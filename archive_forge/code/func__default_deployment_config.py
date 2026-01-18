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
def _default_deployment_config(self, create_mode=True):
    config = {'assume_role_arn': self.assumed_role_arn, 'execution_role_arn': None, 'bucket': None, 'image_url': None, 'region_name': self.region_name, 'archive': False, 'instance_type': DEFAULT_SAGEMAKER_INSTANCE_TYPE, 'instance_count': DEFAULT_SAGEMAKER_INSTANCE_COUNT, 'vpc_config': None, 'data_capture_config': None, 'synchronous': True, 'timeout_seconds': 1200, 'variant_name': None, 'env': None, 'tags': None, 'async_inference_config': {}, 'serverless_config': {}}
    if create_mode:
        config['mode'] = DEPLOYMENT_MODE_CREATE
    else:
        config['mode'] = DEPLOYMENT_MODE_REPLACE
    return config