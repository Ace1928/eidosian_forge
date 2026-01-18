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
def _validate_pyenv_is_available():
    """
    Validates pyenv is available. If not, throws an `MlflowException` with a brief instruction on
    how to install pyenv.
    """
    url = 'https://github.com/pyenv/pyenv#installation' if not is_windows() else 'https://github.com/pyenv-win/pyenv-win#installation'
    if not _is_pyenv_available():
        raise MlflowException(f'Could not find the pyenv binary. See {url} for installation instructions.')