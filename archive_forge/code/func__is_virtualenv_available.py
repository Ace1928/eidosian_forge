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
def _is_virtualenv_available():
    """
    Returns True if virtualenv is available, otherwise False.
    """
    return shutil.which('virtualenv') is not None