# Importing pandas, numpy and sklearn
import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Loading Bitcoin data using pandas
dataframe = pd.read_csv("BTC-USD(6Y).csv")
dataframe.drop(['Date'],1,inplace=True)

# Creation of variable for forecasting price of Bitcoin 'n' days into the future
n = 30  #30 days forecast

# Creating target column (dependent variable) shifted 'n' units up
dataframe['Prediction'] = dataframe[['Close']].shift(-n)


# Creating independent dataset by converting dataframe into numpy array and dropping 'Prediction' column
X = np.array(dataframe.drop(['Prediction'],1))
# Removing last 'n' rows where n is the number of prediction days
X = X[:-n]

# Creating dependent or target dataset by converting 'Prediction' column to numpy array
Y = np.array(dataframe['Prediction'])
# Removing last 'n' rows where n is the number of prediction days
Y = Y[:-n]

np.where(dataframe.values >= np.finfo(np.float64).max)

# Split the data into 80% training and 20% testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
# Replacing NaN values to zero, if any
x_train = np.nan_to_num(x_train)
y_train = np.nan_to_num(y_train)
x_test = np.nan_to_num(x_test)
y_test = np.nan_to_num(y_test)


# Select prediction_array equal to the last 30 rows of the original data set from the close column
prediction_array = np.array(dataframe.drop(['Prediction'],1))[-n:]
#print(prediction_array)

# Create and train the Support Vector Machine
rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001) # Create the model
rbf.fit(x_train, y_train) # Train the model

# Model Testing: Return and print accuracy score.
# Accuracy score lies between 0 and 1, 1 being the best possible score.
rbf_confidence = rbf.score(x_test, y_test)
print("svr_rbf accuracy: ", rbf_confidence)

# Making a second test dataset by taking last 30 rows from x_test and y_test where actual prices are known
x_test2 = x_test[-30:]
y_test2 = y_test[-30:]
y_pred = rbf.predict(x_test2)

# Print the predicted value
svm_prediction = rbf.predict(x_test)
print("Predicted price using SVM:")
print(svm_prediction)
print()

#Print the actual values
print("Actual price:")
print(y_test)
print()

# Print the model predictions for the next 'n=30' days
svm_prediction = rbf.predict(prediction_array)
print("Prediction of next 30 days:")
print(svm_prediction)

# Plotting graphs - for forecasting 30 day Bitcoin prices and comparing values of actual y_test2 values and predicted values
y1 = np.array(y_test2)
y2 = np.array(y_pred)
y3 = np.array(svm_prediction)
plt.subplot(2,1,1)
plt.plot(y1)
plt.plot(y2)
plt.title('Actual price and predicted price')
plt.legend(["Actual price","Predicted price"],loc="lower right")
plt.ylabel("Bitcoin Price(USD)")
plt.subplot(2,1,2)
plt.title("30 Day Forecast")
plt.ylabel("Bitcoin Price(USD)")
plt.plot(y3)
plt.legend(["Predicted Values"],loc="lower right")

plt.tight_layout()
plt.show()


