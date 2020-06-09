# -*- coding: utf-8 -*-
"""
Artificial Neural Network for Churn modelling
"""


# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Alternative methodology
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.compose import ColumnTransformer

label_encoder_x_1 = LabelEncoder()
X[: , 2] = label_encoder_x_1.fit_transform(X[:,2])
transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [1]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
X = transformer.fit_transform(X.tolist())
X = X.astype('float64')

#To avoid dummy trap
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#PART 2 : Developing artificial neural network
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout 


#Initializing the Neural Network
classifier = Sequential()

#Adding input layer and first hidden layer / with dropout
classifier.add(Dense(output_dim = 6,init = 'uniform', activation ='relu',
                     input_dim = 11))
classifier.add(Dropout(p=0.1))    

#Adding second hidden layer
classifier.add(Dense(output_dim = 6,init = 'uniform', activation ='relu'))   
classifier.add(Dropout(p=0.1))                  

#Adding the output layer
classifier.add(Dense(output_dim = 1,init = 'uniform', activation ='sigmoid')) 

#Compiling the Artificial Neural Network
classifier.compile(optimizer='adam', loss= 'binary_crossentropy',
                   metrics=['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train,y_train, batch_size=10,
               nb_epoch=100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)


#Making prediction for single value

new_pred = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,
                                         1,1,50000]])))
new_pred=(new_pred>0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Evaluating , improving and tuning a ANN

#Evaluating a ANN
from keras.wrappers.scikit_learn import KerasClassifier         
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def classifier_build():
        classifier = Sequential()
        classifier.add(Dense(output_dim = 6,init = 'uniform', activation ='relu',
                     input_dim = 11))
        classifier.add(Dense(output_dim = 6,init = 'uniform', activation ='relu'))                        
        classifier.add(Dense(output_dim = 1,init = 'uniform', activation ='sigmoid')) 
        classifier.compile(optimizer='adam', loss= 'binary_crossentropy',
                   metrics=['accuracy'])
        return classifier
    
classifier = KerasClassifier(build_fn=classifier_build ,
                             batch_size = 10 ,nb_epoch =100)
accuracies = cross_val_score(estimator = classifier , X =X_train,
                             y= y_train , cv=10,n_jobs=-1)
mean = accuracies.mean()
variance=accuracies.std()

#Evaluating a ANN using GridSearch
from keras.wrappers.scikit_learn import KerasClassifier  
from sklearn.model_selection import GridSearchCV
from keras.layers import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def classifier_build(optimizer):
        classifier = Sequential()
        classifier.add(Dense(output_dim = 6,init = 'uniform', activation ='relu',
                     input_dim = 11))
        classifier.add(Dropout(p=0.1))
        classifier.add(Dense(output_dim = 6,init = 'uniform', activation ='relu'))   
        classifier.add(Dropout(p=0.1))                  
        classifier.add(Dense(output_dim = 1,init = 'uniform', activation ='sigmoid')) 
        classifier.compile(optimizer=optimizer, loss= 'binary_crossentropy',
                   metrics=['accuracy'])
        return classifier
    
classifier = KerasClassifier(build_fn=classifier_build)
#Creatinng a dictionary for parameters to be tuned
param = {'batch_size' :[25,32],
         'nb_epoch' :[100,500],
         'optimizer' :['adam','rmsprop']}         
grid_search= GridSearchCV(estimator=classifier,
                          param_grid=param,
                          scoring='accuracy',
                          cv =10)                   
grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy=grid_search.best_score_





    


























