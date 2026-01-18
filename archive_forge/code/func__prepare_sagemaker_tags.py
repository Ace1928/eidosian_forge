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
def _prepare_sagemaker_tags(config_tags: List[Dict[str, str]], sagemaker_tags: Optional[Dict[str, str]]=None):
    if not sagemaker_tags:
        return config_tags
    if SAGEMAKER_APP_NAME_TAG_KEY in sagemaker_tags:
        raise MlflowException.invalid_parameter_value(f"Duplicate tag provided for '{SAGEMAKER_APP_NAME_TAG_KEY}'")
    parsed = [{'Key': key, 'Value': str(value)} for key, value in sagemaker_tags.items()]
    return config_tags + parsed