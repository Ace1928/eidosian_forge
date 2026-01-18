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
def _complete_multipart_upload(self, run_id, path, upload_id, part_etags):
    return self._call_endpoint(DatabricksMlflowArtifactsService, CompleteMultipartUpload, message_to_json(CompleteMultipartUpload(run_id=run_id, path=path, upload_id=upload_id, part_etags=part_etags)))