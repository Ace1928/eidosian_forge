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
def _find_endpoint(endpoint_name, sage_client):
    """
    Finds a SageMaker endpoint with the specified name in the caller's AWS account, returning a
    NoneType if the endpoint is not found.

    Args:
        sage_client: A boto3 client for SageMaker.

    Returns:
        If the endpoint exists, a dictionary of endpoint attributes. If the endpoint does not
        exist, ``None``.
    """
    endpoints_page = sage_client.list_endpoints(MaxResults=100, NameContains=endpoint_name)
    while True:
        for endpoint in endpoints_page['Endpoints']:
            if endpoint['EndpointName'] == endpoint_name:
                return endpoint
        if 'NextToken' in endpoints_page:
            endpoints_page = sage_client.list_endpoints(MaxResults=100, NextToken=endpoints_page['NextToken'], NameContains=endpoint_name)
        else:
            return None