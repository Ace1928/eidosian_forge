import json
import logging
import math
import os
import posixpath
import urllib.parse
from mimetypes import guess_type
from mlflow.entities import FileInfo
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialInfo
from mlflow.store.artifact.cloud_artifact_repo import (
from mlflow.store.artifact.s3_artifact_repo import _get_s3_client
from mlflow.utils.file_utils import read_chunk
from mlflow.utils.request_utils import cloud_storage_http_request
from mlflow.utils.rest_utils import augmented_raise_for_status
def _get_s3_client(self):
    return _get_s3_client(addressing_style=self._addressing_style, access_key_id=self._access_key_id, secret_access_key=self._secret_access_key, session_token=self._session_token, region_name=self._region_name, s3_endpoint_url=self._s3_endpoint_url)