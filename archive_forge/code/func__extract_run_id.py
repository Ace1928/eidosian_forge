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
@staticmethod
def _extract_run_id(artifact_uri):
    """
        The artifact_uri is expected to be
        dbfs:/databricks/mlflow-tracking/<EXP_ID>/<RUN_ID>/artifacts/<path>
        Once the path from the input uri is extracted and normalized, it is
        expected to be of the form
        databricks/mlflow-tracking/<EXP_ID>/<RUN_ID>/artifacts/<path>

        Hence the run_id is the 4th element of the normalized path.

        Returns:
            run_id extracted from the artifact_uri.
        """
    artifact_path = extract_and_normalize_path(artifact_uri)
    return artifact_path.split('/')[3]