import datetime
import importlib.metadata
import os
import posixpath
import urllib.parse
from collections import namedtuple
from packaging.version import Version
from mlflow.entities import FileInfo
from mlflow.entities.multipart_upload import (
from mlflow.environment_variables import (
from mlflow.exceptions import _UnsupportedMultipartUploadException
from mlflow.store.artifact.artifact_repo import ArtifactRepository, MultipartUploadMixin
from mlflow.utils.file_utils import relative_path_to_artifact_path
@staticmethod
def _gcs_mpu_arguments(filename: str, blob) -> GCSMPUArguments:
    """See :py:func:`google.cloud.storage.transfer_manager.upload_chunks_concurrently`"""
    from google.cloud.storage.transfer_manager import _headers_from_metadata
    bucket = blob.bucket
    client = blob.client
    transport = blob._get_transport(client)
    hostname = client._connection.get_api_base_url_for_mtls()
    url = f'{hostname}/{bucket.name}/{blob.name}'
    base_headers, object_metadata, content_type = blob._get_upload_arguments(client, None, filename=filename, command='tm.upload_sharded')
    headers = {**base_headers, **_headers_from_metadata(object_metadata)}
    if blob.user_project is not None:
        headers['x-goog-user-project'] = blob.user_project
    if blob.kms_key_name is not None and 'cryptoKeyVersions' not in blob.kms_key_name:
        headers['x-goog-encryption-kms-key-name'] = blob.kms_key_name
    return GCSMPUArguments(transport=transport, url=url, headers=headers, content_type=content_type)