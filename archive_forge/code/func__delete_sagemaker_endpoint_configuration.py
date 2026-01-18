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
def _delete_sagemaker_endpoint_configuration(endpoint_config_name, sage_client):
    """
    Args:
        sage_client: A boto3 client for SageMaker.

    Returns:
        ARN of the deleted endpoint configuration.
    """
    endpoint_config_info = sage_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
    sage_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    return endpoint_config_info['EndpointConfigArn']