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
def _create_sagemaker_model(model_name, model_s3_path, model_uri, flavor, vpc_config, image_url, execution_role, sage_client, env, tags):
    """
    Args:
        model_name: The name to assign the new SageMaker model that is created.
        model_s3_path: S3 path where the model artifacts are stored.
        model_uri: URI of the MLflow model associated with the new SageMaker model.
        flavor: The name of the flavor of the model.
        vpc_config: A dictionary specifying the VPC configuration to use when creating the
            new SageMaker model associated with this SageMaker endpoint.
        image_url: URL of the ECR-hosted Docker image that will serve as the
            model's container,
        execution_role: The ARN of the role that SageMaker will assume when creating the model.
        sage_client: A boto3 client for SageMaker.
        env: A dictionary of environment variables to set for the model.
        tags: A dictionary of tags to apply to the SageMaker model.

    Returns:
        AWS response containing metadata associated with the new model.
    """
    tags['model_uri'] = str(model_uri)
    create_model_args = {'ModelName': model_name, 'PrimaryContainer': {'Image': image_url, 'ModelDataUrl': model_s3_path, 'Environment': _get_deployment_config(flavor_name=flavor, env_override=env)}, 'ExecutionRoleArn': execution_role, 'Tags': [{'Key': key, 'Value': str(value)} for key, value in tags.items()]}
    if vpc_config is not None:
        create_model_args['VpcConfig'] = vpc_config
    return sage_client.create_model(**create_model_args)