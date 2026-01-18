import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

# Load the dataset
data = pd.read_csv("/home/lloyd/Downloads/exampledata/example.csv")

# Separate numeric and categorical columns
numeric_data = data.select_dtypes(include=[np.number])
categorical_data = data.select_dtypes(exclude=[np.number])

# Handle missing values for numeric data
imputer = SimpleImputer(strategy="mean")
numeric_data_imputed = imputer.fit_transform(numeric_data)

# Handle missing values for categorical data if necessary
# For example, using the most frequent strategy
imputer_cat = SimpleImputer(strategy="most_frequent")
categorical_data_imputed = imputer_cat.fit_transform(categorical_data)

# Encode categorical variables
encoder = OneHotEncoder(handle_unknown="ignore")
categorical_data_encoded = encoder.fit_transform(categorical_data_imputed)

# Combine numeric and encoded categorical data
data_imputed = np.hstack((numeric_data_imputed, categorical_data_encoded.toarray()))

# Scale numerical features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

# Create the preprocessed dataset
preprocessed_data = pd.DataFrame(data_scaled)
