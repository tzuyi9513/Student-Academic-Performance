import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Lasso, Ridge, LassoCV, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from dmba import classificationSummary, regressionSummary
from sklearn.ensemble import RandomForestClassifier

from dmba import classificationSummary, gainsChart, liftChart
from dmba import regressionSummary, exhaustive_search
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import adjusted_r2_score, AIC_score, BIC_score
from dmba.metric import AIC_score

pd.options.mode.chained_assignment = None 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

PREDICTORS = ['Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'sex_M',
'address_U', 'famsize_LE3', 'Pstatus_T','reason_home', 'reason_other',
'reason_reputation', 'guardian_mother', 'guardian_other',
'schoolsup_yes', 'famsup_yes', 'paid_yes', 'activities_yes',
'nursery_yes', 'higher_yes', 'internet_yes', 'romantic_yes']

PATH = os.getcwd() + '/grapihcs/'
if not os.path.exists(PATH):
    os.mkdir(PATH)

def clean_data(fname):
    '''
    To sum up, we choose to keep Medu, Fedu, Walc, Dalc, G1, G2, and G3 in our analysis. 
    As for grades, we choose to convert 3 grades into average grade to do further analysis.
    '''
    df = pd.read_csv(fname)

    #We will drop the variables that are not valuable for our anaysis.
    df.drop(columns=['school','age','Fjob','Mjob'],inplace=True)

    #convert the data types
    for col in df.columns:
        if df[col].dtypes =='object':
            df[col] = df[col].astype('category')

    df[df.isna()].count() #There is no missing value in the dataset 
    df[df.duplicated()] # There is no duplicated value in the dataset 

    #Convert average_grade to binary.
    df['average_grade'] = (df['G1'] + df['G2'] + df['G3']) / 3
    df['grade_pass'] = df.apply(lambda row: 1 if row['average_grade'] >= 10 else 0, axis=1)

    return df

def plot_corr_heatmap(df):
    fname = 'heatmap.png'
    destination = os.path.join(PATH, fname)
    print('(0) Save ' + fname + ': correlation heatmap\n')\
    # Compute the correlation matrix
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    _f, _ax = plt.subplots(figsize=(40, 40))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    _heat_map = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5, vmin=-.5,center=0, annot = True,
                square=True, linewidths=.5, cbar_kws={'shrink': .8})

    plt.savefig(destination)

def plot_corr_btw_fedu_n_medu(df):
    '''
    FEdu & Medu
    In this correlation chart, we find the correlation efficient between 'Fedu' and 'Medu' was high, 
    and we can take a closer look at the reltionship. 
    From the following bar chart, we can see Father's education had a different distribution compared to mother's education. 
    Most father's education were 2, while mothers tended to have higher education level.
    Therefore, we choose to keep both FEdu and MEdu.
    '''
    fname = 'fedu & medu.png.png'
    destination = os.path.join(PATH, fname)

    print('(1) Save ' + fname + ': correlation between FEdu and Medu\n')\

    # Bar charts of FEdu & Medu
    plt.rc('figure', figsize=(10, 5))

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)
    fedu = df['Fedu'].value_counts()
    fedu = pd.DataFrame(fedu)
    ax1.set_title('Father\'s Education Distribution')
    plt.ylabel('count of Fedu')
    plt.xlabel('Fedu')
    ax1.bar(fedu.index, fedu['Fedu'], color = 'cornflowerblue')

    ax2 = fig.add_subplot(1, 2, 2)
    medu = df['Medu'].value_counts()
    medu = pd.DataFrame(medu)
    ax2.set_title('Mother\'s Education Distribution')
    plt.ylabel('count of Medu')
    plt.xlabel('Medu')
    ax2.bar(medu.index, medu['Medu'], color = 'pink')

    plt.savefig(destination)

def plot_corr_btw_walc_n_dalc(df):
    '''
    Walc & Dalc
    Also, we find the correlation coefficient between 'Walc' and 'Dalc' was high. 
    We used the bar chart to visualize their distribution, 
        and found although both Walc and Dalc had an increase pattern, 
    the distribution seemed to be a little different. 
    After level 2, the Dalc dropped dramatically than Walc.
    Therefore, we chose to keep both variables.
    '''
    fname = 'walc & dalc.png'
    destination = os.path.join(os.getcwd() + '/grapihcs/', fname)

    print('(2) Save ' + fname + ': correlation between walc and dalc\n')\

    # Bar charts of Walc & Dalc
    plt.rc('figure', figsize=(10, 5))
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    walc = df['Walc'].value_counts()
    walc = pd.DataFrame(walc)
    ax1.set_title('Weekly Alchocol Consumption Distribution')
    plt.ylabel('Count of Walc')
    plt.xlabel('Walc')
    ax1.bar(walc.index, walc['Walc'], color = 'Brown')

    ax2 = fig.add_subplot(1, 2, 2)
    dalc = df['Dalc'].value_counts()
    dalc = pd.DataFrame(dalc)
    ax2.set_title('Daily Alchocol Consumption Distribution')
    plt.ylabel('Count of Dalc')
    plt.xlabel('Dalc')
    ax2.bar(dalc.index, dalc['Dalc'], color = 'indianred')

    plt.savefig(destination)

def plot_corr_among_grades(df):
    '''
    G1, G2 and G3
    The 3 grades seemed to have high correlation. 
    We visualzed their distribution to see it clearly. 
    We found most students falled around 10 points in the three grades. 
    However, more students got 0 in G2 and G3, while G1 had fewer students with 0 points.
    The 3 grades distributions were different, so we chose to keep all of them.
    '''
    fname = 'correlation among grades.png.png'
    destination = os.path.join(PATH, fname)

    print('(3) Save ' + fname + ': correlation among grades: G1, G2, and G3\n')

    # Bar charts of G1, G2 and G3
    plt.rc('figure', figsize=(15, 5))
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    g1 = df['G1'].value_counts()
    g1 = pd.DataFrame(g1)
    ax1.set_title('G1 Distribution')
    plt.ylabel('Count of G1')
    plt.xlabel('G1')
    ax1.bar(g1.index, g1['G1'], color = 'turquoise')

    ax2 = fig.add_subplot(1, 3, 2)
    g2 = df['G2'].value_counts()
    g2 = pd.DataFrame(g2)
    ax2.set_title('G2 Distribution')
    plt.ylabel('Count of G2')
    plt.xlabel('G2')
    ax2.bar(g2.index, g2['G2'], color = 'lightseagreen')

    ax3 = fig.add_subplot(1, 3, 3)
    g3 = df['G3'].value_counts()
    g3 = pd.DataFrame(g3)
    ax3.set_title('G3 Distribution')
    plt.ylabel('Count of G3')
    plt.xlabel('G3')
    ax3.bar(g3.index, g3['G3'], color = 'mediumaquamarine')

    plt.savefig(destination)

def plot_corre_btw_medu_n_avg_grade(df):
    '''
    Compute the Medu's counts and the average_grade based on the Medu's groups
    '''
    fname = 'avg grade & medu.png'
    destination = os.path.join(PATH, fname)

    print('(4) Save ' + fname + ': correlation between Medu and Average Grade\n')

    mean_medu = df.groupby('Medu').mean()['average_grade'].values.tolist()
    del mean_medu[0]  # neglect the first element

    count_medu = df.groupby('Medu').count()['average_grade'].values.tolist()
    del count_medu[0]  # neglect the first element

    x = ['1', '2', '3', '4']

    _fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Medu')
    ax1.set_ylabel('Average Grade', color=color)
    ax1.plot(x, mean_medu, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('frequency', color=color)  # we already handled the x-label with ax1
    ax2.bar(x, count_medu, 0.4, color=color, alpha = 0.8)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Correlation between Average Grade and Medu')

    plt.savefig(destination)

def plot_corre_btw_fedu_n_avg_grade(df):
    '''
    Compute the Fedu's counts and the average_grade based on the Fedu's groups
    '''
    fname = 'avg grade & fedu.png'
    destination = os.path.join(PATH, fname)

    print('(5) Save ' + fname + ': correlation between Medu and Average Grade\n')

    mean_fedu = df.groupby('Fedu').mean()['average_grade'].values.tolist()
    del mean_fedu[0]  # neglect the first element

    count_fedu = df.groupby('Fedu').count()['average_grade'].values.tolist()
    del count_fedu[0]  # neglect the first element

    x = ['1', '2', '3', '4']

    _fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Fedu')
    ax1.set_ylabel('Average Grade', color=color)
    ax1.plot(x, mean_fedu, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('frequency', color=color)  # we already handled the x-label with ax1
    ax2.bar(x, count_fedu, 0.4, color=color, alpha = 0.8)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Correlation between Average Grade and Fedu')
    plt.savefig(destination)

def plot_corr_study_time_n_avg_grade(df):
    fname = 'studytime & avg grade.png'
    destination = os.path.join(PATH, fname)

    print('(6) Save ' + fname + ': correlation between  and Average Grade\n')

    mean_studytime = df.groupby('studytime').mean()['average_grade'].values.tolist()
    count_studytime = df.groupby('studytime').count()['average_grade'].values.tolist()

    _fig, ax1 = plt.subplots()
    x = ['1', '2', '3', '4']
    color = 'tab:red'
    ax1.set_xlabel('studytime')
    ax1.set_ylabel('Average Grade', color=color)
    ax1.plot(x, mean_studytime, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('frequency', color=color)  # we already handled the x-label with ax1
    ax2.bar(x, count_studytime, 0.4, color=color, alpha = 0.8)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Correlation between Average Grade and studytime')

    plt.savefig(destination)

def plot_corr_btw_travel_time_n_avg_grade(df):
    ''''
    Compute the traveltime's counts and the average_grade based on the traveltime's groups
    '''
    fname = 'travel_time & avg_grade.png'
    destination = os.path.join(PATH, fname)

    print('(7) Save ' + fname + ': correlation between Travel Time and Average Grade\n')

    mean_traveltime = df.groupby('traveltime').mean()['average_grade'].values.tolist()
    count_traveltime = df.groupby('traveltime').count()['average_grade'].values.tolist()

    x = ['1', '2', '3', '4']

    _fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Travel Time')
    ax1.set_ylabel('Average Grade', color=color)
    ax1.plot(x, mean_traveltime, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('frequency', color=color)  # we already handled the x-label with ax1
    ax2.bar(x, count_traveltime, 0.4, color=color, alpha = 0.8)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Correlation between Average Grade and Travel Time')

    plt.savefig(destination)

def partition_data(df):
    df = pd.get_dummies(df, drop_first=True)
    x = df[PREDICTORS]
    y = df['grade_pass']
    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.2, random_state=1)

    return train_x, valid_x, train_y, valid_y

def display_logistic_regression(train_x, valid_x, train_y, valid_y):
    print('(8) Display logistic regression\n')
    # fit a logistic regression (set penalty=l2 and C=1e42 to avoid regularization)
    logit_reg = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')
    logit_reg.fit(train_x, train_y)

    print('intercept ', logit_reg.intercept_[0])
    print(pd.DataFrame({'coeff': sorted(abs(logit_reg.coef_[0]), reverse=True)}, index=PREDICTORS), '\n')
    print('AIC', AIC_score(valid_y, logit_reg.predict(valid_x), df=len(train_x.columns) + 1))

    classificationSummary(train_y, logit_reg.predict(train_x))
    classificationSummary(valid_y, logit_reg.predict(valid_x))

    prediction_valid = logit_reg.predict(valid_x)
    prediction_train = logit_reg.predict(train_x)

    print('precision on test is:', precision_score(valid_y,prediction_valid))
    print('recall on test is:', recall_score(valid_y,prediction_valid))
    print('f1 on test is:', f1_score(valid_y,prediction_valid))
    print('Logistic Regression:Accuracy on train is:', accuracy_score(train_y,prediction_train))
    print('Logistic Regression:Accuracy on test is:', accuracy_score(valid_y,prediction_valid), '\n')

def display_decision_tree(train_x, valid_x, train_y, valid_y):
    print('(9) Display Decision Tree\n')

    fname = 'decision tree.png'
    destination = os.path.join(PATH, fname)

    fullClassTree = DecisionTreeClassifier(max_depth=4, random_state = 1)
    fullClassTree.fit(train_x, train_y)

    plt.figure()
    plot_tree(fullClassTree)
    plt.savefig(destination)


    prediction_train = fullClassTree.predict(train_x)#use the DT model to predict on the training data
    prediction_valid = fullClassTree.predict(valid_x)#use the DT model to predict on the validation data

    print('precision on test is:',precision_score(valid_y,prediction_valid))
    print('recall on test is:',recall_score(valid_y,prediction_valid))
    print('f1 on test is:',f1_score(valid_y,prediction_valid))
    print('Logistic Regression:Accuracy on train is:',accuracy_score(train_y,prediction_train))
    print('Logistic Regression:Accuracy on test is:',accuracy_score(valid_y,prediction_valid), '\n')

    importances = fullClassTree.feature_importances_
    important_df = pd.DataFrame({'feature': train_x.columns, 'importance': importances})#,'std':std})
    important_df = important_df.sort_values('importance',ascending=False)
    print(important_df)


def main():
    fname = 'student-mat.csv'

    df = clean_data(fname)
    train_x, valid_x, train_y, valid_y = partition_data(df)

    plot_corr_heatmap(df)
    plot_corr_btw_fedu_n_medu(df)
    plot_corr_btw_walc_n_dalc(df)
    plot_corr_among_grades(df)
    plot_corre_btw_medu_n_avg_grade(df)
    plot_corre_btw_fedu_n_avg_grade(df)
    plot_corr_btw_travel_time_n_avg_grade(df)

    display_logistic_regression(train_x, valid_x, train_y, valid_y)
    display_decision_tree(train_x, valid_x, train_y, valid_y)

if __name__ == '__main__':
    main()