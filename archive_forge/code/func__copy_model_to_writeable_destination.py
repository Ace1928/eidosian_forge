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
def _copy_model_to_writeable_destination(model_src, dst):
    """
    Copies the specified `model_src` directory, which may be read-only, to the writeable `dst`
    directory.
    """
    os.makedirs(dst, exist_ok=True)
    for model_item in os.listdir(model_src):
        copy_fn = shutil.copytree if os.path.isdir(model_item) else shutil.copy2
        copy_fn(src=os.path.join(model_src, model_item), dst=os.path.join(dst, model_item))