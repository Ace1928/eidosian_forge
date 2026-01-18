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
def _get_cluster_mlflow_run_cmd(project_dir, run_id, entry_point, parameters, env_manager):
    cmd = ['mlflow', 'run', project_dir, '--entry-point', entry_point]
    if env_manager:
        cmd += ['--env-manager', env_manager]
    mlflow_run_arr = list(map(quote, cmd))
    if run_id:
        mlflow_run_arr.extend(['-c', json.dumps({MLFLOW_LOCAL_BACKEND_RUN_ID_CONFIG: run_id})])
    if parameters:
        for key, value in parameters.items():
            mlflow_run_arr.extend(['-P', f'{key}={value}'])
    return mlflow_run_arr