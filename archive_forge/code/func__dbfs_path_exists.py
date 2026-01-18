import hashlib
import json
import logging
import os
import posixpath
import re
import tempfile
import textwrap
import time
from shlex import quote
from mlflow import tracking
from mlflow.entities import RunStatus
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID, MLFLOW_TRACKING_URI
from mlflow.exceptions import ExecutionException, MlflowException
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.projects.utils import MLFLOW_LOCAL_BACKEND_RUN_ID_CONFIG
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils import databricks_utils, file_utils, rest_utils
from mlflow.utils.mlflow_tags import (
from mlflow.utils.uri import is_databricks_uri, is_http_uri
from mlflow.version import VERSION, is_release_version
def _dbfs_path_exists(self, dbfs_path):
    """
        Return True if the passed-in path exists in DBFS for the workspace corresponding to the
        default Databricks CLI profile. The path is expected to be a relative path to the DBFS root
        directory, e.g. 'path/to/file'.
        """
    host_creds = databricks_utils.get_databricks_host_creds(self.databricks_profile_uri)
    response = rest_utils.http_request(host_creds=host_creds, endpoint='/api/2.0/dbfs/get-status', method='GET', json={'path': f'/{dbfs_path}'})
    try:
        json_response_obj = json.loads(response.text)
    except Exception:
        raise MlflowException(f'API request to check existence of file at DBFS path {dbfs_path} failed with status code {response.status_code}. Response body: {response.text}')
    error_code_field = 'error_code'
    if error_code_field in json_response_obj:
        if json_response_obj[error_code_field] == 'RESOURCE_DOES_NOT_EXIST':
            return False
        raise ExecutionException(f'Got unexpected error response when checking whether file {dbfs_path} exists in DBFS: {json_response_obj}')
    return True