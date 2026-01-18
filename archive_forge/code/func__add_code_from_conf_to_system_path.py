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
def _add_code_from_conf_to_system_path(local_path, conf, code_key=FLAVOR_CONFIG_CODE):
    """Checks if any code_paths were logged with the model in the flavor conf and prepends
    the directory to the system path.

    Args:
        local_path: The local path containing model artifacts.
        conf: The flavor-specific conf that should contain the FLAVOR_CONFIG_CODE
            key, which specifies the directory containing custom code logged as artifacts.
        code_key: The key used by the flavor to indicate custom code artifacts.
            By default this is FLAVOR_CONFIG_CODE.
    """
    assert isinstance(conf, dict), '`conf` argument must be a dict.'
    if code_key in conf and conf[code_key]:
        code_path = os.path.join(local_path, conf[code_key])
        _add_code_to_system_path(code_path)