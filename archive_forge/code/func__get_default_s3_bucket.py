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
def _get_default_s3_bucket(region_name, **assume_role_credentials):
    import boto3
    sess = boto3.Session()
    account_id = _get_account_id(**assume_role_credentials)
    bucket_name = f'{DEFAULT_BUCKET_NAME_PREFIX}-{region_name}-{account_id}'
    s3 = sess.client('s3', **assume_role_credentials)
    response = s3.list_buckets()
    buckets = [b['Name'] for b in response['Buckets']]
    if bucket_name not in buckets:
        _logger.info('Default bucket `%s` not found. Creating...', bucket_name)
        bucket_creation_kwargs = {'ACL': 'bucket-owner-full-control', 'Bucket': bucket_name}
        if region_name != 'us-east-1':
            bucket_creation_kwargs['CreateBucketConfiguration'] = {'LocationConstraint': region_name}
        response = s3.create_bucket(**bucket_creation_kwargs)
        _logger.info('Bucket creation response: %s', response)
    else:
        _logger.info('Default bucket `%s` already exists. Skipping creation.', bucket_name)
    return bucket_name