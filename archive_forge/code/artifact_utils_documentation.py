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
Copy the artifacts from ``source`` to the destination Databricks workspace (DBFS) given by
    ``databricks_profile_uri`` or the current tracking URI.

    Args:
        source: Source location for the artifacts to copy.
        run_id: Run ID to associate the artifacts with.
        source_host_uri: Specifies the source artifact's host URI (e.g. Databricks tracking URI)
            if applicable. If not given, defaults to the current tracking URI.
        target_databricks_profile_uri: Specifies the destination Databricks host. If not given,
            defaults to the current tracking URI.

    Returns:
        The DBFS location in the target Databricks workspace the model files have been
        uploaded to.
    