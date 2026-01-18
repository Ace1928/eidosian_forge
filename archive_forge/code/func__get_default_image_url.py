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
def _get_default_image_url(region_name):
    import boto3
    if (env_img := MLFLOW_SAGEMAKER_DEPLOY_IMG_URL.get()):
        return env_img
    ecr_client = boto3.client('ecr', region_name=region_name)
    repository_conf = ecr_client.describe_repositories(repositoryNames=[DEFAULT_IMAGE_NAME])['repositories'][0]
    return (repository_conf['repositoryUri'] + ':{version}').format(version=mlflow.version.VERSION)