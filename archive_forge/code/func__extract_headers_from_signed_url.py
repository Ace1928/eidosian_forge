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
def _extract_headers_from_signed_url(self, headers):
    filtered_headers = filter(lambda h: 'name' in h and 'value' in h, headers)
    return {header.get('name'): header.get('value') for header in filtered_headers}