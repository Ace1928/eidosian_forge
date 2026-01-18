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
def _assume_role_and_get_credentials(assume_role_arn=None):
    """
    Assume a new role in AWS and return the credentials for that role.
    When ``assume_role_arn`` is ``None`` or an empty string,
    this function does nothing and returns an empty dictionary.

    Args:
        assume_role_arn: Optional ARN of the role that will be assumed

    Returns:
        Dict with credentials of the assumed role
    """
    import boto3
    if not assume_role_arn:
        return {}
    sts_client = boto3.client('sts')
    sts_response = sts_client.assume_role(RoleArn=assume_role_arn, RoleSessionName='mlflow-sagemaker')
    _logger.info('Assuming role %s for deployment!', assume_role_arn)
    return {'aws_access_key_id': sts_response['Credentials']['AccessKeyId'], 'aws_secret_access_key': sts_response['Credentials']['SecretAccessKey'], 'aws_session_token': sts_response['Credentials']['SessionToken']}