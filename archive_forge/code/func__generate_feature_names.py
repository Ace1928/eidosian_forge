import importlib
import logging
import os
import sys
import time
import cloudpickle
from packaging.version import Version
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.recipes.artifacts import DataframeArtifact, TransformerArtifact
from mlflow.recipes.cards import BaseCard
from mlflow.recipes.step import BaseStep, StepClass
from mlflow.recipes.utils.execution import get_step_output_path
from mlflow.recipes.utils.step import get_pandas_data_profiles, validate_classification_config
from mlflow.recipes.utils.tracking import TrackingConfig, get_recipe_tracking_config
def _generate_feature_names(num_features):
    max_length = len(str(num_features))
    return ['f_' + str(i).zfill(max_length) for i in range(num_features)]