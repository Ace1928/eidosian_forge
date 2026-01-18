import logging
import os
import posixpath
import re
import shutil
from typing import Any, Dict, Optional
import pandas as pd
import yaml
from packaging.version import Version
import mlflow
from mlflow import environment_variables, mleap, pyfunc
from mlflow.environment_variables import MLFLOW_DFS_TMP
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature, infer_signature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _LOG_MODEL_INFER_SIGNATURE_WARNING_TEMPLATE
from mlflow.models.utils import _Example, _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.databricks_artifact_repo import DatabricksArtifactRepository
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import (
from mlflow.utils import _get_fully_qualified_class_name, databricks_utils
from mlflow.utils.autologging_utils import autologging_integration, safe_patch
from mlflow.utils.class_utils import _get_class_from_string
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import (
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.uri import (
@classmethod
def _try_file_exists(cls, dfs_path):
    try:
        return cls._fs().exists(dfs_path)
    except Exception as ex:
        _logger.debug('Unexpected exception while checking if model uri is visible on DFS: %s', ex)
    return False