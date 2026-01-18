import json
import logging
import pathlib
import shutil
import tempfile
import uuid
from typing import Any, Dict, Optional
import mlflow
from mlflow.environment_variables import MLFLOW_RUN_CONTEXT
from mlflow.exceptions import MlflowException, RestException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.recipes.utils import get_recipe_name
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.context.registry import resolve_tags
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.tracking.fluent import set_experiment as fluent_set_experiment
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.file_utils import path_to_local_file_uri, path_to_local_sqlite_uri
from mlflow.utils.git_utils import get_git_branch, get_git_commit, get_git_repo_url
from mlflow.utils.mlflow_tags import (
def _get_run_name(run_name_prefix):
    if run_name_prefix is None:
        return None
    sep = '-'
    num = uuid.uuid4().hex[:8]
    return f'{run_name_prefix}{sep}{num}'