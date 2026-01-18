import datetime
import importlib
import logging
import os
import re
import shutil
import sys
import warnings
import cloudpickle
import yaml
import mlflow
from mlflow.entities import SourceType, ViewType
from mlflow.environment_variables import MLFLOW_RECIPES_EXECUTION_TARGET_STEP_NAME
from mlflow.exceptions import BAD_REQUEST, INVALID_PARAMETER_VALUE, MlflowException
from mlflow.models import Model
from mlflow.recipes.artifacts import (
from mlflow.recipes.cards import BaseCard
from mlflow.recipes.step import BaseStep, StepClass
from mlflow.recipes.utils.execution import get_step_output_path
from mlflow.recipes.utils.metrics import (
from mlflow.recipes.utils.step import (
from mlflow.recipes.utils.tracking import (
from mlflow.recipes.utils.wrapped_recipe_model import WrappedRecipeModel
from mlflow.tracking import MlflowClient
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.databricks_utils import get_databricks_env_vars, get_databricks_run_url
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import (
from mlflow.utils.string_utils import strip_prefix
def _label_encoded_fitted_estimator(self, estimator, X_train, y_train):
    from sklearn.preprocessing import LabelEncoder
    label_encoder = None
    target_column_class_labels = None
    encoded_y_train = y_train
    if 'classification' in self.recipe:
        label_encoder = LabelEncoder()
        label_encoder.fit(y_train)
        target_column_class_labels = label_encoder.classes_
        import pandas as pd
        encoded_y_train = pd.Series(label_encoder.transform(y_train))

    def inverse_label_encoder(predicted_output):
        if not label_encoder:
            return predicted_output
        return label_encoder.inverse_transform(predicted_output)
    estimator.fit(X_train, encoded_y_train)
    original_predict = estimator.predict

    def wrapped_predict(*args, **kwargs):
        return inverse_label_encoder(original_predict(*args, **kwargs))
    estimator.predict = wrapped_predict
    return (estimator, {'target_column_class_labels': target_column_class_labels})