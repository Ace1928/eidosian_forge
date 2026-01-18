from yellowbrick.classifier import ClassificationReport
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
X_train = ...  # Training features
y_train = ...  # Training labels
X_test = ...  # Testing features
y_test = ...  # Testing labels

# Train a random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create a classification report visualizer
visualizer = ClassificationReport(model, classes=["Class 0", "Class 1"])

# Fit the visualizer to the testing data
visualizer.fit(X_test, y_test)

# Display the visualization
visualizer.show()
