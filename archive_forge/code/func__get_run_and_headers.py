import base64
import functools
import logging
import os
import shutil
from contextlib import contextmanager
import mlflow
from mlflow.entities import Run
from mlflow.environment_variables import MLFLOW_UNITY_CATALOG_PRESIGNED_URLS_ENABLED
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_service_pb2 import UcModelRegistryService
from mlflow.protos.service_pb2 import GetRun, MlflowService
from mlflow.store.artifact.presigned_url_artifact_repo import PresignedUrlArtifactRepository
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry.rest_store import BaseRestStore
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils._unity_catalog_utils import (
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_host_creds, is_databricks_uri
from mlflow.utils.mlflow_tags import (
from mlflow.utils.proto_json_utils import message_to_json, parse_dict
from mlflow.utils.rest_utils import (
def _get_run_and_headers(self, run_id):
    if run_id is None or not is_databricks_uri(self.tracking_uri):
        return (None, None)
    host_creds = self.get_tracking_host_creds()
    endpoint, method = _TRACKING_METHOD_TO_INFO[GetRun]
    response = http_request(host_creds=host_creds, endpoint=endpoint, method=method, params={'run_id': run_id})
    try:
        verify_rest_response(response, endpoint)
    except MlflowException:
        _logger.warning(f"Unable to fetch model version's source run (with ID {run_id}) from tracking server. The source run may be deleted or inaccessible to the current user. No run link will be recorded for the model version.")
        return (None, None)
    headers = response.headers
    js_dict = response.json()
    parsed_response = GetRun.Response()
    parse_dict(js_dict=js_dict, message=parsed_response)
    run = Run.from_proto(parsed_response.run)
    return (headers, run)