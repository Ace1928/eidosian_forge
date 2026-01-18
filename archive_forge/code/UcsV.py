import pandas as pd
from faker import Faker

# Load the dataset
data = pd.read_csv("dataset.csv")

# Create a Faker instance
faker = Faker()

# Define the columns to anonymize
columns_to_anonymize = ["name", "email", "phone"]

# Anonymize the specified columns
for column in columns_to_anonymize:
    data[column] = data[column].apply(
        lambda x: (
            faker.name()
            if column == "name"
            else faker.email() if column == "email" else faker.phone_number()
        )
    )

# Save the anonymized dataset
data.to_csv("anonymized_dataset.csv", index=False)
