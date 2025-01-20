#! C:\Users\vishw\Desktop\Projects\Publishing\myenv\Scripts\python.exe
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
#Importing of all the libraries

inputData=pd.read_csv(r'C:\Users\vishw\Desktop\Projects\Publishing\hippocampus1.csv') #Import aand saving the dataset in a variable
print(inputData.isna().sum(), end='\n\n\n') #Finding and printing null values in the dataset
print(inputData[inputData.isna().any(axis=1)], end='\n\n\n') #Finding and printing the corresponding row containing null value(s)
reference_genes=pd.read_csv(r'C:\Users\vishw\Desktop\Projects\Publishing\universaldataset3.csv')


#                                                     DATA PRE-PROCESSING

inputData.drop(columns=['t', 'B', 'Gene.title'], axis=1).to_csv('finalDatah1.csv', index=False) #Removal of undesirable columns ans saving the rest of the data in a new CSV file
finalDatah1=pd.read_csv(r'C:\Users\vishw\Desktop\Projects\Publishing\finalDatah1.csv') #Importing and saving the new dataset
print(finalDatah1.isnull().sum(), end='\n\n\n')
print(finalDatah1.dtypes, end='\n\n\n') #Check the data types of the columns in the dataset
reference_genes_set=set(reference_genes['Gene.symbol'])

X = finalDatah1[['logFC', 'P.Value', 'adj.P.Val']] #Features variable
y_str = finalDatah1['Gene.symbol'] #Target variable

label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(y_str) #Conversion of string to numeric

X_filtered = X[(X['adj.P.Val'] < 0.5) & (X['logFC'].abs() > 1)] #Threshold restriction
Y_filtered = Y[X_filtered.index]

X_train, X_test, y_train, y_test = train_test_split(X_filtered, Y_filtered, test_size=0.15, random_state=44) #Division of dataset into test ad train dataset

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) #Scaling of train dataset
X_test_scaled = scaler.transform(X_test) #Scaling of test dataset


#                                                            LASSO MODEL

lasso_model = LogisticRegression(penalty='l1', solver='liblinear', random_state=44) #Lasso model configuration

print("The Lasso Model predictions are displayed below ->", end='\n\n')

lasso_model.fit(X_train_scaled, y_train) #Model training

y_pred = lasso_model.predict(X_test_scaled) #Model prediction

y_pred_str = label_encoder.inverse_transform(y_pred.round().astype(int)) #Conversion of numeric to string

print(y_pred_str, end='\n') #Printing the predictions
print(len(y_pred_str), end='\n\n\n') #Printing the size of predictions


#                                                        RANDOM FOREST MODEL

RF = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=44) #RF Model configuration

print('The Random Forest Model predictions are displayed below ->', end='\n\n')

RF.fit(X_train_scaled, y_train) #Model training

yRFPredictions = RF.predict(X_test_scaled) #Model prediction

yRFDecodedPredictions = label_encoder.inverse_transform(yRFPredictions.round().astype(int)) #conversion of numeric to string

print(yRFDecodedPredictions, end='\n') #Printing the predictions
print(len(yRFDecodedPredictions), end='\n\n\n') #Printing the size of predictions


#                                                            SVM-RFE MODEL

svmModel = SVC(kernel='linear') #SVM configuration
rfe = RFE(svmModel, n_features_to_select=5) #RFE configuration

print('The SVM-RFE Model predictions are displayed below ->', end='\n\n')

rfe.fit(X_train_scaled, y_train) #RFE training

xTrainSelected = rfe.transform(X_train_scaled)
xTestSelected = rfe.transform(X_test_scaled)

svmModel.fit(xTrainSelected, y_train) #SVM training

ySVMRFEPredictions = svmModel.predict(xTestSelected) #Model prediction
ySVMRFEDecodedPredictions = label_encoder.inverse_transform(ySVMRFEPredictions.round().astype(int)) #Convrsion of numeric to string

print(ySVMRFEDecodedPredictions, end='\n') #Printing the predictions
print(len(ySVMRFEDecodedPredictions), end='\n\n\n') #Printing the size of predictions


#                                                      Overlapping Genes Check

lassoSet = set(y_pred_str)
randomForestSet = set(yRFDecodedPredictions)
svmrfeSet = set(ySVMRFEDecodedPredictions) #Saving the predictions all the models in a set

overlappingGenes = randomForestSet.intersection(svmrfeSet, lassoSet) #Printing the overlapping genes
print("Overlapping Genes:\n", overlappingGenes)
print(len(overlappingGenes)) #Printing the size of overlapping genes

# Convert all gene identifiers to strings, remove leading/trailing spaces, and make case-insensitive
reference_genes_set = {str(gene).strip() for gene in reference_genes_set}
lassoSet = {str(gene).strip() for gene in lassoSet}
randomForestSet = {str(gene).strip() for gene in randomForestSet}
svmrfeSet = {str(gene).strip() for gene in svmrfeSet}

print("Lasso Predictions (sample):", list(lassoSet)[:10])  # Display first 10 predictions
print("Random Forest Predictions (sample):", list(randomForestSet)[:10])
print("SVM-RFE Predictions (sample):", list(svmrfeSet)[:10])
print("Reference Genes (sample):", list(reference_genes_set)[:10])

accuracy = lambda predicted_set, reference_set: len(predicted_set.intersection(reference_set)) / len(predicted_set) if len(predicted_set) > 0 else 0

# Calculate overlap accuracy
lasso_accuracy = accuracy(lassoSet, reference_genes_set)
rf_accuracy = accuracy(randomForestSet, reference_genes_set)
svmrfe_accuracy = accuracy(svmrfeSet, reference_genes_set)

print(f"Lasso Model Accuracy: {lasso_accuracy}")
print(f"Random Forest Model Accuracy: {rf_accuracy}")
print(f"SVM-RFE Model Accuracy: {svmrfe_accuracy}")