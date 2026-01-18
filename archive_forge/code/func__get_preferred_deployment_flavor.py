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
def _get_preferred_deployment_flavor(model_config):
    """
    Obtains the flavor that MLflow would prefer to use when deploying the model.
    If the model does not contain any supported flavors for deployment, an exception
    will be thrown.

    Args:
        model_config: An MLflow model object

    Returns:
        The name of the preferred deployment flavor for the specified model
    """
    if mleap.FLAVOR_NAME in model_config.flavors:
        return mleap.FLAVOR_NAME
    elif pyfunc.FLAVOR_NAME in model_config.flavors:
        return pyfunc.FLAVOR_NAME
    else:
        raise MlflowException(message='The specified model does not contain any of the supported flavors for deployment. The model contains the following flavors: {model_flavors}. Supported flavors: {supported_flavors}'.format(model_flavors=model_config.flavors.keys(), supported_flavors=SUPPORTED_DEPLOYMENT_FLAVORS), error_code=RESOURCE_DOES_NOT_EXIST)