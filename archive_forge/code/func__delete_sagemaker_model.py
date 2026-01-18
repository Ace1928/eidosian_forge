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
def _delete_sagemaker_model(model_name, sage_client, s3_client):
    """
    Args:
        sage_client: A boto3 client for SageMaker.
        s3_client: A boto3 client for S3.

    Returns:
        ARN of the deleted model.
    """
    model_info = sage_client.describe_model(ModelName=model_name)
    model_arn = model_info['ModelArn']
    model_data_url = model_info['PrimaryContainer']['ModelDataUrl']
    parsed_data_url = urllib.parse.urlparse(model_data_url)
    bucket_name = parsed_data_url.netloc
    bucket_key = parsed_data_url.path.lstrip('/')
    s3_client.delete_object(Bucket=bucket_name, Key=bucket_key)
    sage_client.delete_model(ModelName=model_name)
    return model_arn