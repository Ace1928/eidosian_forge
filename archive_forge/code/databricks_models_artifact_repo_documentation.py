import json
import logging
import os
import posixpath
import mlflow.tracking
from mlflow.entities import FileInfo
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.utils.models import (
from mlflow.utils.databricks_utils import (
from mlflow.utils.file_utils import (
from mlflow.utils.rest_utils import http_request
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri

    Performs storage operations on artifacts controlled by a Databricks-hosted model registry.

    Signed access URIs for the appropriate cloud storage locations are fetched from the
    MLflow service and used to download model artifacts.

    The artifact_uri is expected to be of the form
    - `models:/<model_name>/<model_version>`
    - `models:/<model_name>/<stage>`  (refers to the latest model version in the given stage)
    - `models:/<model_name>/latest`  (refers to the latest of all model versions)
    - `models://<profile>/<model_name>/<model_version or stage or 'latest'>`

    Note : This artifact repository is meant is to be instantiated by the ModelsArtifactRepository
    when the client is pointing to a Databricks-hosted model registry.
    