import os
import posixpath
import re
import urllib.parse
from typing import List
import requests
from mlflow.azure.client import patch_adls_file_upload, patch_adls_flush, put_adls_file_creation
from mlflow.entities import FileInfo
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialInfo
from mlflow.store.artifact.cloud_artifact_repo import (
def _retryable_adls_function(self, func, artifact_file_path, **kwargs):
    try:
        func(**kwargs)
    except requests.HTTPError as e:
        if e.response.status_code in [403]:
            new_credentials = self._get_write_credential_infos([artifact_file_path])[0]
            kwargs['sas_url'] = new_credentials.signed_uri
            func(**kwargs)
        else:
            raise e