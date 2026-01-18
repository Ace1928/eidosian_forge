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
def _get_overridden_pyfunc_model_config(pyfunc_config: Dict[str, Any], load_config: Dict[str, Any], logger) -> Dict[str, Any]:
    """
    Updates the inference configuration according to the model's configuration and the overrides.
    Only arguments already present in the inference configuration can be updated. The environment
    variable ``MLFLOW_PYFUNC_INFERENCE_CONFIG`` can also be used to provide additional inference
    configuration.
    """
    overrides = {}
    if (env_overrides := os.getenv('MLFLOW_PYFUNC_INFERENCE_CONFIG')):
        logger.debug('Inference configuration is being loaded from ``MLFLOW_PYFUNC_INFERENCE_CONFIG`` environ.')
        overrides.update(dict(json.loads(env_overrides)))
    if load_config:
        overrides.update(load_config)
    if not overrides:
        return pyfunc_config
    if not pyfunc_config:
        logger.warning(f"Argument(s) {', '.join(overrides.keys())} were ignored since the model's ``pyfunc`` flavor doesn't accept model configuration. Use ``model_config`` when logging the model to allow it.")
        return None
    valid_keys = set(pyfunc_config.keys()) & set(overrides.keys())
    ignored_keys = set(overrides.keys()) - valid_keys
    allowed_config = {key: overrides[key] for key in valid_keys}
    if ignored_keys:
        logger.warning(f'Argument(s) {', '.join(ignored_keys)} were ignored since they are not valid keys in the corresponding section of the ``pyfunc`` flavor. Use ``model_config`` when logging the model to include the keys you plan to indicate. Current allowed configuration includes {', '.join(pyfunc_config.keys())}')
    pyfunc_config.update(allowed_config)
    return pyfunc_config