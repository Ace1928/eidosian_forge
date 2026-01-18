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
def _get_flavor_configuration_from_ml_model_file(ml_model_file, flavor_name):
    model_conf = Model.load(ml_model_file)
    if flavor_name not in model_conf.flavors:
        raise MlflowException(f'Model does not have the "{flavor_name}" flavor', RESOURCE_DOES_NOT_EXIST)
    return model_conf.flavors[flavor_name]