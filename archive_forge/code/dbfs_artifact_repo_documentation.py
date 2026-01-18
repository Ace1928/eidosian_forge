import json
import os
import posixpath
import mlflow.utils.databricks_utils
from mlflow.entities import FileInfo
from mlflow.environment_variables import MLFLOW_ENABLE_DBFS_FUSE_ARTIFACT_REPO
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.databricks_artifact_repo import DatabricksArtifactRepository
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracking._tracking_service import utils
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.file_utils import relative_path_to_artifact_path
from mlflow.utils.rest_utils import RESOURCE_DOES_NOT_EXIST, http_request, http_request_safe
from mlflow.utils.string_utils import strip_prefix
from mlflow.utils.uri import (

    Returns an ArtifactRepository subclass for storing artifacts on DBFS.

    This factory method is used with URIs of the form ``dbfs:/<path>``. DBFS-backed artifact
    storage can only be used together with the RestStore.

    In the special case where the URI is of the form
    `dbfs:/databricks/mlflow-tracking/<Exp-ID>/<Run-ID>/<path>',
    a DatabricksArtifactRepository is returned. This is capable of storing access controlled
    artifacts.

    Args:
        artifact_uri: DBFS root artifact URI.

    Returns:
        Subclass of ArtifactRepository capable of storing artifacts on DBFS.
    