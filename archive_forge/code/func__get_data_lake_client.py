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
def _get_data_lake_client(account_url, credential):
    from azure.storage.filedatalake import DataLakeServiceClient
    return DataLakeServiceClient(account_url, credential)