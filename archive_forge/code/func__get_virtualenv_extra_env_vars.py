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
def _get_virtualenv_extra_env_vars(env_root_dir=None):
    extra_env = {'PIP_NO_INPUT': '1'}
    if env_root_dir is not None:
        extra_env['PIP_CACHE_DIR'] = os.path.join(env_root_dir, _PIP_CACHE_DIR)
    return extra_env