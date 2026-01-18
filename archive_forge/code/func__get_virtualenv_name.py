import logging
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from packaging.version import Version
import mlflow
from mlflow.environment_variables import MLFLOW_ENV_ROOT
from mlflow.exceptions import MlflowException
from mlflow.models.model import MLMODEL_FILE_NAME, Model
from mlflow.utils.conda import _PIP_CACHE_DIR
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.environment import (
from mlflow.utils.file_utils import remove_on_error
from mlflow.utils.os import is_windows
from mlflow.utils.process import _exec_cmd, _join_commands
from mlflow.utils.requirements_utils import _parse_requirements
def _get_virtualenv_name(python_env, work_dir_path, env_id=None):
    requirements = _parse_requirements(python_env.dependencies, is_constraint=False, base_dir=work_dir_path)
    return _get_mlflow_env_name(str(python_env) + ''.join(map(str, sorted(requirements))) + (env_id or ''))