import logging
import os
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.openai_utils import (
from mlflow.utils.rest_utils import augmented_raise_for_status
from mlflow.deployments import BaseDeploymentClient
def _check_openai_key():
    if 'OPENAI_API_KEY' not in os.environ:
        raise MlflowException('OPENAI_API_KEY environment variable not set', error_code=INVALID_PARAMETER_VALUE)