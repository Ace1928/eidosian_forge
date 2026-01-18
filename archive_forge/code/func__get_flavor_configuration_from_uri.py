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
def _get_flavor_configuration_from_uri(model_uri, flavor_name, logger):
    """Obtains the configuration for the specified flavor from the specified
    MLflow model uri. If the model does not contain the specified flavor,
    an exception will be thrown.

    Args:
        model_uri: The path to the root directory of the MLflow model for which to load
            the specified flavor configuration.
        flavor_name: The name of the flavor configuration to load.
        logger: The local flavor's logger to report the resolved path of the model uri.

    Returns:
        The flavor configuration as a dictionary.
    """
    try:
        resolved_uri = model_uri
        if RunsArtifactRepository.is_runs_uri(model_uri):
            resolved_uri = RunsArtifactRepository.get_underlying_uri(model_uri)
            logger.info("'%s' resolved as '%s'", model_uri, resolved_uri)
        elif ModelsArtifactRepository.is_models_uri(model_uri):
            resolved_uri = ModelsArtifactRepository.get_underlying_uri(model_uri)
            logger.info("'%s' resolved as '%s'", model_uri, resolved_uri)
        try:
            ml_model_file = _download_artifact_from_uri(artifact_uri=append_to_uri_path(resolved_uri, MLMODEL_FILE_NAME))
        except Exception:
            logger.debug(f'Failed to download an "{MLMODEL_FILE_NAME}" model file from resolved URI {resolved_uri}. Falling back to downloading from original model URI {model_uri}', exc_info=True)
            ml_model_file = get_artifact_repository(artifact_uri=model_uri).download_artifacts(artifact_path=MLMODEL_FILE_NAME)
    except Exception as ex:
        raise MlflowException(f'Failed to download an "{MLMODEL_FILE_NAME}" model file from "{model_uri}"', RESOURCE_DOES_NOT_EXIST) from ex
    return _get_flavor_configuration_from_ml_model_file(ml_model_file, flavor_name)