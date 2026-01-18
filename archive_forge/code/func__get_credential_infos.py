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
def _get_credential_infos(self, request_message_class, run_id, paths):
    """
        Issue one or more requests for artifact credentials, providing read or write
        access to the specified run-relative artifact `paths` within the MLflow Run specified
        by `run_id`. The type of access credentials, read or write, is specified by
        `request_message_class`.

        Args:
            paths: The specified run-relative artifact paths within the MLflow Run.
            run_id: The specified MLflow Run.
            request_message_class: Specifies the type of access credentials, read or write.

        Returns:
            A list of `ArtifactCredentialInfo` objects providing read access to the specified
            run-relative artifact `paths` within the MLflow Run specified by `run_id`.
        """
    credential_infos = []
    for paths_chunk in chunk_list(paths, _MAX_CREDENTIALS_REQUEST_SIZE):
        page_token = None
        while True:
            json_body = message_to_json(request_message_class(run_id=run_id, path=paths_chunk, page_token=page_token))
            response = self._call_endpoint(DatabricksMlflowArtifactsService, request_message_class, json_body)
            credential_infos += response.credential_infos
            page_token = response.next_page_token
            if not page_token or len(response.credential_infos) == 0:
                break
    return credential_infos