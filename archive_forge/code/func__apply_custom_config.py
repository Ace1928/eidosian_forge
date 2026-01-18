import json
import logging
import os
import platform
import signal
import sys
import tarfile
import time
import urllib.parse
from subprocess import Popen
from typing import Any, Dict, List, Optional
import mlflow
import mlflow.version
from mlflow import mleap, pyfunc
from mlflow.deployments import BaseDeploymentClient, PredictionsResponse
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.container import (
from mlflow.models.container import (
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import get_unique_resource_id
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import dump_input_data
def _apply_custom_config(self, config, custom_config):
    int_fields = {'instance_count', 'timeout_seconds'}
    bool_fields = {'synchronous', 'archive'}
    dict_fields = {'vpc_config', 'data_capture_config', 'tags', 'env', 'async_inference_config', 'serverless_config'}
    for key, value in custom_config.items():
        if key not in config:
            continue
        if key in int_fields and (not isinstance(value, int)):
            value = int(value)
        elif key in bool_fields and (not isinstance(value, bool)):
            value = value == 'True'
        elif key in dict_fields and (not isinstance(value, dict)):
            value = json.loads(value)
        config[key] = value