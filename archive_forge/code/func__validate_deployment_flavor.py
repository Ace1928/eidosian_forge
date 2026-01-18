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
def _validate_deployment_flavor(model_config, flavor):
    """
    Checks that the specified flavor is a supported deployment flavor
    and is contained in the specified model. If one of these conditions
    is not met, an exception is thrown.

    Args:
        model_config: An MLflow Model object
        flavor: The deployment flavor to validate
    """
    if flavor not in SUPPORTED_DEPLOYMENT_FLAVORS:
        raise MlflowException(message=f'The specified flavor: `{flavor}` is not supported for deployment. Please use one of the supported flavors: {SUPPORTED_DEPLOYMENT_FLAVORS}', error_code=INVALID_PARAMETER_VALUE)
    elif flavor not in model_config.flavors:
        raise MlflowException(message=f'The specified model does not contain the specified deployment flavor: `{flavor}`. Please use one of the following deployment flavors that the model contains: {model_config.flavors.keys()}', error_code=RESOURCE_DOES_NOT_EXIST)