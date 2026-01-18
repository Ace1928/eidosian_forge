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
def _resolve_estimator_plugin(self, plugin_str, X_train, y_train, output_directory):
    plugin_str = plugin_str.replace('/', '.')
    estimator_fn = importlib.import_module(f'mlflow.recipes.steps.{plugin_str}').get_estimator_and_best_params
    estimator, best_parameters = estimator_fn(X_train, y_train, self.task, self.extended_task, self.step_config, self.recipe_root, self.evaluation_metrics, self.primary_metric)
    self.best_estimator_name = estimator.__class__.__name__
    self.best_estimator_class = f'{estimator.__class__.__module__}.{estimator.__class__.__name__}'
    self.best_parameters = best_parameters
    self._write_best_parameters_outputs(output_directory, automl_params=best_parameters)
    return estimator