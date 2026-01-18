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
def _safe_dump_with_numeric_values(self, data, file, **kwargs):
    import numpy as np
    processed_data = {}
    for key, value in data.items():
        if isinstance(value, np.floating):
            processed_data[key] = float(value)
        elif isinstance(value, np.integer):
            processed_data[key] = int(value)
        elif isinstance(value, dict):
            processed_data[key] = str(value)
        else:
            processed_data[key] = str(value)
    if len(processed_data) > 0:
        yaml.safe_dump(processed_data, file, **kwargs)