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
def _upload_s3(local_model_path, bucket, prefix, region_name, s3_client, **assume_role_credentials):
    """
    Upload dir to S3 as .tar.gz.

    Args:
        local_model_path: Local path to a dir.
        bucket: S3 bucket where to store the data.
        prefix: Path within the bucket.
        region_name: The AWS region in which to upload data to S3.
        s3_client: A boto3 client for S3.

    Returns:
        S3 path of the uploaded artifact.
    """
    import boto3
    sess = boto3.Session(region_name=region_name, **assume_role_credentials)
    with TempDir() as tmp:
        model_data_file = tmp.path('model.tar.gz')
        _make_tarfile(model_data_file, local_model_path)
        with open(model_data_file, 'rb') as fobj:
            key = os.path.join(prefix, 'model.tar.gz')
            obj = sess.resource('s3').Bucket(bucket).Object(key)
            obj.upload_fileobj(fobj)
            response = s3_client.put_object_tagging(Bucket=bucket, Key=key, Tagging={'TagSet': [{'Key': 'SageMaker', 'Value': 'true'}]})
            _logger.info('tag response: %s', response)
            return f's3://{bucket}/{key}'