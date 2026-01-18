from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Load the dataset
X = ...  # Features
y = ...  # Target variable

# Create a logistic regression model
model = LogisticRegression()

# Create an RFE object
rfe = RFE(estimator=model, n_features_to_select=10)

# Fit the RFE object to the data
rfe.fit(X, y)

# Get the selected features
selected_features = X.columns[rfe.support_]

# Print the selected features
print("Selected Features:", selected_features)
