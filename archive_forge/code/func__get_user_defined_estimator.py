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
def _get_user_defined_estimator(self, X_train, y_train, validation_df, run, output_directory):
    sys.path.append(self.recipe_root)
    estimator_fn = getattr(importlib.import_module(_USER_DEFINED_TRAIN_STEP_MODULE), self.step_config['estimator_method'])
    estimator_hardcoded_params = self.step_config['estimator_params']
    if self.using_rebalancing:
        estimator_hardcoded_params['class_weight'] = self.original_class_weights
    if self.step_config['tuning_enabled']:
        estimator_hardcoded_params, best_hp_params = self._tune_and_get_best_estimator_params(run.info.run_id, estimator_hardcoded_params, estimator_fn, X_train, y_train, validation_df)
        best_combined_params = dict(estimator_hardcoded_params, **best_hp_params)
        estimator = estimator_fn(best_combined_params)
        all_estimator_params = estimator.get_params()
        default_params_keys = all_estimator_params.keys() - best_combined_params.keys()
        default_params = {k: all_estimator_params[k] for k in default_params_keys}
        self._write_best_parameters_outputs(output_directory, best_hp_params=best_hp_params, best_hardcoded_params=estimator_hardcoded_params, default_params=default_params)
    elif len(estimator_hardcoded_params) > 0:
        estimator = estimator_fn(estimator_hardcoded_params)
        all_estimator_params = estimator.get_params()
        default_params_keys = all_estimator_params.keys() - estimator_hardcoded_params.keys()
        default_params = {k: all_estimator_params[k] for k in default_params_keys}
        self._write_best_parameters_outputs(output_directory, best_hardcoded_params=estimator_hardcoded_params, default_params=default_params)
    else:
        estimator = estimator_fn()
        default_params = estimator.get_params()
        self._write_best_parameters_outputs(output_directory, default_params=default_params)
    return estimator