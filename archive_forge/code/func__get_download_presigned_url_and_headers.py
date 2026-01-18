import json
import os
import posixpath
from mlflow.entities import FileInfo
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialInfo
from mlflow.protos.databricks_filesystem_service_pb2 import (
from mlflow.store.artifact.cloud_artifact_repo import CloudArtifactRepository
from mlflow.utils.file_utils import download_file_using_http_uri
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.request_utils import augmented_raise_for_status, cloud_storage_http_request
from mlflow.utils.rest_utils import (
def _get_download_presigned_url_and_headers(self, remote_file_path):
    remote_file_full_path = posixpath.join(self.artifact_uri, remote_file_path)
    endpoint, method = FILESYSTEM_METHOD_TO_INFO[CreateDownloadUrlRequest]
    req_body = message_to_json(CreateDownloadUrlRequest(path=remote_file_full_path))
    response_proto = CreateDownloadUrlResponse()
    return call_endpoint(host_creds=self.db_creds, endpoint=endpoint, method=method, json_body=req_body, response_proto=response_proto)