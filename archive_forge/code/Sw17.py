import pandas as pd

# Load the dataset
data = pd.read_csv("dataset.csv")

# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:")
print(missing_values)

# Check for duplicates
duplicates = data.duplicated().sum()
print(f"\nNumber of Duplicates: {duplicates}")

# Check data type consistency
data_types = data.dtypes
print("\nData Types:")
print(data_types)

# Check for outliers
numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns
for column in numeric_columns:
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    print(f"\nOutliers in {column}:")
    print(outliers)
