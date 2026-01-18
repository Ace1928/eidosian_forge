from typing import TYPE_CHECKING, Any, Dict, List, Optional
import requests
from mlflow import MlflowException
from mlflow.deployments import BaseDeploymentClient
from mlflow.deployments.constants import (
from mlflow.deployments.server.config import Endpoint
from mlflow.deployments.server.constants import (
from mlflow.deployments.utils import resolve_endpoint_url
from mlflow.environment_variables import (
from mlflow.protos.databricks_pb2 import BAD_REQUEST
from mlflow.store.entities.paged_list import PagedList
from mlflow.utils.annotations import experimental
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.rest_utils import augmented_raise_for_status, http_request
from mlflow.utils.uri import join_paths
def _call_endpoint(self, method: str, route: str, json_body: Optional[str]=None, timeout: Optional[int]=None):
    call_kwargs = {}
    if method.lower() == 'get':
        call_kwargs['params'] = json_body
    else:
        call_kwargs['json'] = json_body
    response = http_request(host_creds=get_default_host_creds(self.target_uri), endpoint=route, method=method, timeout=MLFLOW_HTTP_REQUEST_TIMEOUT.get() if timeout is None else timeout, retry_codes=MLFLOW_DEPLOYMENT_CLIENT_REQUEST_RETRY_CODES, raise_on_status=False, **call_kwargs)
    augmented_raise_for_status(response)
    return response.json()