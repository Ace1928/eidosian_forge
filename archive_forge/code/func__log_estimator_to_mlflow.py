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
def _log_estimator_to_mlflow(self, estimator, X_train_sampled, on_worker=False):
    from mlflow.models import infer_signature
    if hasattr(estimator, 'best_score_') and type(estimator.best_score_) in [int, float]:
        mlflow.log_metric('best_cv_score', estimator.best_score_)
    if hasattr(estimator, 'best_params_'):
        mlflow.log_params(estimator.best_params_)
    if on_worker:
        mlflow.log_params(estimator.get_params())
        estimator_tags = {'estimator_name': estimator.__class__.__name__, 'estimator_class': estimator.__class__.__module__ + '.' + estimator.__class__.__name__}
        mlflow.set_tags(estimator_tags)
    estimator_schema = infer_signature(X_train_sampled, estimator.predict(X_train_sampled.copy()))
    return mlflow.sklearn.log_model(estimator, f'{self.name}/estimator', signature=estimator_schema, code_paths=self.code_paths)