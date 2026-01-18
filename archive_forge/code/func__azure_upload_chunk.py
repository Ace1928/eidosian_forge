import base64
import logging
import os
import posixpath
import uuid
import requests
import mlflow.tracking
from mlflow.azure.client import (
from mlflow.entities import FileInfo
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_artifacts_pb2 import (
from mlflow.protos.databricks_pb2 import (
from mlflow.protos.service_pb2 import GetRun, ListArtifacts, MlflowService
from mlflow.store.artifact.cloud_artifact_repo import (
from mlflow.utils import chunk_list
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.file_utils import (
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.request_utils import cloud_storage_http_request
from mlflow.utils.rest_utils import (
from mlflow.utils.uri import (
def _azure_upload_chunk(self, credentials, headers, local_file, artifact_file_path, start_byte, size):
    block_id = base64.b64encode(uuid.uuid4().hex.encode()).decode('utf-8')
    chunk = read_chunk(local_file, size, start_byte)
    try:
        put_block(credentials.signed_uri, block_id, chunk, headers=headers)
    except requests.HTTPError as e:
        if e.response.status_code in [401, 403]:
            _logger.info('Failed to authorize request, possibly due to credential expiration. Refreshing credentials and trying again...')
            credential_info = self._get_write_credential_infos([artifact_file_path])[0]
            put_block(credential_info.signed_uri, block_id, chunk, headers=headers)
        else:
            raise e
    return block_id