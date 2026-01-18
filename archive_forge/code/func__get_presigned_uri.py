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
def _get_presigned_uri(self, artifact_file_path):
    """
        Gets the presigned URL required to upload a file to or download a file from a given Azure
        storage location.

        Args:
            artifact_file_path: Path of the file relative to the artifact repository root.

        Returns:
            a string presigned URL.
        """
    sas_token = self.credential.signature
    return f'https://{self.account_name}.{self.domain_suffix}/{self.container}/{self.base_data_lake_directory}/{artifact_file_path}?{sas_token}'