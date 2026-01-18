import os
import pathlib
import posixpath
import tempfile
import urllib.parse
import uuid
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.dbfs_artifact_repo import DbfsRestArtifactRepository
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.os import is_windows
from mlflow.utils.uri import add_databricks_profile_info_to_artifact_uri, append_to_uri_path
def _get_root_uri_and_artifact_path(artifact_uri):
    """Parse the artifact_uri to get the root_uri and artifact_path.

    Args:
        artifact_uri: The *absolute* URI of the artifact.
    """
    if os.path.exists(artifact_uri):
        if not is_windows():
            root_uri = os.path.dirname(artifact_uri)
            artifact_path = os.path.basename(artifact_uri)
            return (root_uri, artifact_path)
        else:
            artifact_uri = path_to_local_file_uri(artifact_uri)
    parsed_uri = urllib.parse.urlparse(str(artifact_uri))
    prefix = ''
    if parsed_uri.scheme and (not parsed_uri.path.startswith('/')):
        prefix = parsed_uri.scheme + ':'
        parsed_uri = parsed_uri._replace(scheme='')
    if ModelsArtifactRepository.is_models_uri(artifact_uri):
        root_uri, artifact_path = ModelsArtifactRepository.split_models_uri(artifact_uri)
    else:
        artifact_path = posixpath.basename(parsed_uri.path)
        parsed_uri = parsed_uri._replace(path=posixpath.dirname(parsed_uri.path))
        root_uri = prefix + urllib.parse.urlunparse(parsed_uri)
    return (root_uri, artifact_path)