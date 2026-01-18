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
def _get_all_flavor_configurations(model_path):
    """Obtains all the flavor configurations from the specified MLflow model path.

    Args:
        model_path: The path to the root directory of the MLflow model for which to load
            the specified flavor configuration.

    Returns:
        The dictionary contains all flavor configurations with flavor name as key.

    """
    model_configuration_path = os.path.join(model_path, MLMODEL_FILE_NAME)
    if not os.path.exists(model_configuration_path):
        raise MlflowException(f'Could not find an "{MLMODEL_FILE_NAME}" configuration file at "{model_path}"', RESOURCE_DOES_NOT_EXIST)
    model_conf = Model.load(model_configuration_path)
    return model_conf.flavors