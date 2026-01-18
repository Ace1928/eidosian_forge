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
def _get_signed_download_uri(self, path=None):
    if not path:
        path = ''
    json_body = self._make_json_body(path)
    response = self._call_endpoint(json_body, REGISTRY_ARTIFACT_PRESIGNED_URI_ENDPOINT)
    try:
        json_response = json.loads(response.text)
    except ValueError:
        raise MlflowException(f'API request to get presigned uri to for file under path `{path}` failed with status code {response.status_code}. Response body: {response.text}')
    return (json_response.get('signed_uri', None), json_response.get('headers', None))