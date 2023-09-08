# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your dataset
# Replace 'your_dataset.csv' with the actual file path or URL of your dataset
data = pd.read_csv('C:\\Users\\Administrator\\Desktop\\your_dataset.csv')

# Display the first few rows of your dataset to understand its structure
data.head()

# Assume 'price' is the target variable (the variable we want to predict)
# and 'bedrooms', 'sqft', and other relevant features are the predictors
X = data[['bedrooms', 'sqft', 'other_feature_1', 'other_feature_2']]  # Replace with your actual features
y = data['price']  # Replace with your target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Visualize the predictions (optional)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()
