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
def _create_sagemaker_endpoint(endpoint_name, model_name, model_s3_path, model_uri, image_url, flavor, instance_type, vpc_config, data_capture_config, instance_count, role, sage_client, variant_name=None, async_inference_config=None, serverless_config=None, env=None, tags=None):
    """
    Args:
        endpoint_name: The name of the SageMaker endpoint to create.
        model_name: The name to assign the new SageMaker model that will be associated with the
            specified endpoint.
        model_s3_path: S3 path where we stored the model artifacts.
        model_uri: URI of the MLflow model to associate with the specified SageMaker endpoint.
        image_url: URL of the ECR-hosted docker image the model is being deployed into.
        flavor: The name of the flavor of the model to use for deployment.
        instance_type: The type of SageMaker ML instance on which to deploy the model.
        instance_count: The number of SageMaker ML instances on which to deploy the model.
        vpc_config: A dictionary specifying the VPC configuration to use when creating the
            new SageMaker model associated with this SageMaker endpoint.
        data_capture_config: A dictionary specifying the data capture configuration to use when
            creating the new SageMaker model associated with this application.
        role: SageMaker execution ARN role.
        sage_client: A boto3 client for SageMaker.
        variant_name: The name to assign to the new production variant.
        env: A dictionary of environment variables to set for the model.
        tags: A dictionary of tags to apply to the endpoint.
    """
    _logger.info('Creating new endpoint with name: %s ...', endpoint_name)
    model_response = _create_sagemaker_model(model_name=model_name, model_s3_path=model_s3_path, model_uri=model_uri, flavor=flavor, vpc_config=vpc_config, image_url=image_url, execution_role=role, sage_client=sage_client, env=env or {}, tags=tags or {})
    _logger.info('Created model with arn: %s', model_response['ModelArn'])
    if not variant_name:
        variant_name = model_name
    production_variant = {'VariantName': variant_name, 'ModelName': model_name, 'InitialVariantWeight': 1}
    if serverless_config:
        production_variant['ServerlessConfig'] = serverless_config
    else:
        production_variant['InstanceType'] = instance_type
        production_variant['InitialInstanceCount'] = instance_count
    config_name = _get_sagemaker_config_name(endpoint_name)
    config_tags = _get_sagemaker_config_tags(endpoint_name)
    tags_list = _prepare_sagemaker_tags(config_tags, tags)
    endpoint_config_kwargs = {'EndpointConfigName': config_name, 'ProductionVariants': [production_variant], 'Tags': config_tags}
    if async_inference_config:
        endpoint_config_kwargs['AsyncInferenceConfig'] = async_inference_config
    if data_capture_config is not None:
        endpoint_config_kwargs['DataCaptureConfig'] = data_capture_config
    endpoint_config_response = sage_client.create_endpoint_config(**endpoint_config_kwargs)
    _logger.info('Created endpoint configuration with arn: %s', endpoint_config_response['EndpointConfigArn'])
    endpoint_response = sage_client.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name, Tags=tags_list or [])
    _logger.info('Created endpoint with arn: %s', endpoint_response['EndpointArn'])

    def status_check_fn():
        endpoint_info = _find_endpoint(endpoint_name=endpoint_name, sage_client=sage_client)
        if endpoint_info is None:
            return _SageMakerOperationStatus.in_progress('Waiting for endpoint to be created...')
        endpoint_status = endpoint_info['EndpointStatus']
        if endpoint_status == 'Creating':
            return _SageMakerOperationStatus.in_progress(f'Waiting for endpoint to reach the "InService" state. Current endpoint status: "{endpoint_status}"')
        elif endpoint_status == 'InService':
            return _SageMakerOperationStatus.succeeded('The SageMaker endpoint was created successfully.')
        else:
            failure_reason = endpoint_info.get('FailureReason', 'An unknown SageMaker failure occurred. Please see the SageMaker console logs for more information.')
            return _SageMakerOperationStatus.failed(failure_reason)

    def cleanup_fn():
        pass
    return _SageMakerOperation(status_check_fn=status_check_fn, cleanup_fn=cleanup_fn)