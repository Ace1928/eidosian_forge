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
def _parse_abfss_uri(uri):
    """
    Parse an ABFSS URI in the format
    "abfss://<file_system>@<account_name>.<domain_suffix>/<path>",
    returning a tuple consisting of the filesystem, account name, domain suffix, and path

    See more details about ABFSS URIs at
    https://learn.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-abfs-driver#uri-scheme-to-reference-data.
    Also, see different domain suffixes for:
    * Azure China: https://learn.microsoft.com/en-us/azure/china/resources-developer-guide
    * Azure Government: https://learn.microsoft.com/en-us/azure/azure-government/compare-azure-government-global-azure#guidance-for-developers
    * Azure Private Link: https://learn.microsoft.com/en-us/azure/private-link/private-endpoint-dns#government
    Args:
        uri: ABFSS URI to parse

    Returns:
        A tuple containing the name of the filesystem, account name, domain suffix, and path
    """
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != 'abfss':
        raise MlflowException(f'Not an ABFSS URI: {uri}')
    match = re.match('([^@]+)@([^.]+)\\.(.*)', parsed.netloc)
    if match is None:
        raise MlflowException('ABFSS URI must be of the form abfss://<filesystem>@<account>.<domain_suffix>')
    filesystem = match.group(1)
    account_name = match.group(2)
    domain_suffix = match.group(3)
    path = parsed.path
    if path.startswith('/'):
        path = path[1:]
    return (filesystem, account_name, domain_suffix, path)