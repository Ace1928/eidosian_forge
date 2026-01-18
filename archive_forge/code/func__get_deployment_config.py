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
def _get_deployment_config(flavor_name, env_override=None):
    """
    Returns:
        The deployment configuration as a dictionary
    """
    deployment_config = {MLFLOW_DEPLOYMENT_FLAVOR_NAME.name: flavor_name, SERVING_ENVIRONMENT: SAGEMAKER_SERVING_ENVIRONMENT}
    if env_override:
        deployment_config.update(env_override)
    if os.getenv('http_proxy') is not None:
        deployment_config.update({'http_proxy': os.environ['http_proxy']})
    if os.getenv('https_proxy') is not None:
        deployment_config.update({'https_proxy': os.environ['https_proxy']})
    if os.getenv('no_proxy') is not None:
        deployment_config.update({'no_proxy': os.environ['no_proxy']})
    return deployment_config