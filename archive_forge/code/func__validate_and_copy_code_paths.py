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
def _validate_and_copy_code_paths(code_paths, path, default_subpath='code'):
    """Validates that a code path is a valid list and copies the code paths to a directory. This
    can later be used to log custom code as an artifact.

    Args:
        code_paths: A list of files or directories containing code that should be logged
            as artifacts.
        path: The local model path.
        default_subpath: The default directory name used to store code artifacts.
    """
    _validate_code_paths(code_paths)
    if code_paths is not None:
        code_dir_subpath = default_subpath
        for code_path in code_paths:
            try:
                _copy_file_or_tree(src=code_path, dst=path, dst_dir=code_dir_subpath)
            except OSError as e:
                example = ', such as Databricks Notebooks' if is_in_databricks_runtime() else ''
                raise MlflowException(message=(f"Failed to copy the specified code path '{code_path}' into the model artifacts. It appears that your code path includes file(s) that cannot be copied{example}. Please specify a code path that does not include such files and try again.",), error_code=INVALID_PARAMETER_VALUE) from e
    else:
        code_dir_subpath = None
    return code_dir_subpath