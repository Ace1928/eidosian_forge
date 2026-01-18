import json
import logging
import os
import posixpath
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import mlflow
from mlflow import mleap, pyfunc
from mlflow.environment_variables import MLFLOW_DFS_TMP
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.spark import (
from mlflow.store.artifact.databricks_artifact_repo import DatabricksArtifactRepository
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import (
from mlflow.utils import databricks_utils
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import (
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.uri import (
def _validate_env_vars():
    if _JOHNSNOWLABS_ENV_JSON_LICENSE_KEY not in os.environ:
        raise Exception(f'Please set the {_JOHNSNOWLABS_ENV_JSON_LICENSE_KEY} environment variable as the raw license.json string from John Snow Labs')
    _set_env_vars()