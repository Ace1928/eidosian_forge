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
def _azure_adls_gen2_upload_file(self, credentials, local_file, artifact_file_path):
    """
        Uploads a file to a given Azure storage location using the ADLS gen2 API.
        """
    try:
        headers = self._extract_headers_from_credentials(credentials.headers)
        self._retryable_adls_function(func=put_adls_file_creation, artifact_file_path=artifact_file_path, sas_url=credentials.signed_uri, headers=headers)
        futures = {}
        file_size = os.path.getsize(local_file)
        num_chunks = _compute_num_chunks(local_file, MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get())
        use_single_part_upload = num_chunks == 1
        for index in range(num_chunks):
            start_byte = index * MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get()
            future = self.chunk_thread_pool.submit(self._retryable_adls_function, func=patch_adls_file_upload, artifact_file_path=artifact_file_path, sas_url=credentials.signed_uri, local_file=local_file, start_byte=start_byte, size=MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get(), position=start_byte, headers=headers, is_single=use_single_part_upload)
            futures[future] = index
        _, errors = _complete_futures(futures, local_file)
        if errors:
            raise MlflowException(f'Failed to upload at least one part of {artifact_file_path}. Errors: {errors}')
        if not use_single_part_upload:
            self._retryable_adls_function(func=patch_adls_flush, artifact_file_path=artifact_file_path, sas_url=credentials.signed_uri, position=file_size, headers=headers)
    except Exception as err:
        raise MlflowException(err)