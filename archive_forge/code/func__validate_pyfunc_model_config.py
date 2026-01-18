import json
import os
import sys
from typing import Any, Dict
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.protos.databricks_pb2 import (
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.file_utils import _copy_file_or_tree
from mlflow.utils.uri import append_to_uri_path
def _validate_pyfunc_model_config(model_config):
    """
    Validates the values passes in the model_config section. There are no typing
    restrictions but we require them being JSON-serializable.
    """
    if not model_config:
        return
    if not isinstance(model_config, dict) or not all((isinstance(key, str) for key in model_config)):
        raise MlflowException('An invalid ``model_config`` structure was passed. ``model_config`` must be of type ``dict`` with string keys.')
    try:
        json.dumps(model_config)
    except (TypeError, OverflowError):
        raise MlflowException('Values in the provided ``model_config`` are of an unsupported type. Only JSON-serializable data types can be provided as values.')