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
def _create_sagemaker_transform_job(job_name, model_name, model_s3_path, model_uri, image_url, flavor, vpc_config, role, sage_client, s3_client, instance_type, instance_count, s3_input_data_type, s3_input_uri, content_type, compression_type, split_type, s3_output_path, accept, assemble_with, input_filter, output_filter, join_resource):
    """
    Args:
        job_name: Name of the deployed Sagemaker batch transform job.
        model_name: The name to assign the new SageMaker model that will be associated with the
            specified batch transform job.
        model_s3_path: S3 path where we stored the model artifacts.
        model_uri: URI of the MLflow model to associate with the specified SageMaker batch
            transform job.
        image_url: URL of the ECR-hosted docker image the model is being deployed into.
        flavor: The name of the flavor of the model to use for deployment.
        vpc_config: A dictionary specifying the VPC configuration to use when creating the
            new SageMaker model associated with this SageMaker batch transform job.
        role: SageMaker execution ARN role.
        sage_client: A boto3 client for SageMaker.
        s3_client: A boto3 client for S3.
        instance_type: The type of SageMaker ML instance on which to deploy the model.
        instance_count: The number of SageMaker ML instances on which to deploy the model.
        s3_input_data_type: Input data type for the transform job.
        s3_input_uri: S3 key name prefix or a manifest of the input data.
        content_type: The multipurpose internet mail extension (MIME) type of the data.
        compression_type: The compression type of the transform data.
        split_type: The method to split the transform job's data files into smaller batches.
        s3_output_path: The S3 path to store the output results of the Sagemaker transform job.
        accept: The multipurpose internet mail extension (MIME) type of the output data.
        assemble_with: The method to assemble the results of the transform job as a single
            S3 object.
        input_filter: A JSONPath expression used to select a portion of the input data for the
            transform job.
        output_filter: A JSONPath expression used to select a portion of the output data from
            the transform job.
        join_resource: The source of the data to join with the transformed data.
    """
    _logger.info('Creating new batch transform job with name: %s ...', job_name)
    model_response = _create_sagemaker_model(model_name=model_name, model_s3_path=model_s3_path, model_uri=model_uri, flavor=flavor, vpc_config=vpc_config, image_url=image_url, execution_role=role, sage_client=sage_client, env={}, tags={})
    _logger.info('Created model with arn: %s', model_response['ModelArn'])
    transform_input = {'DataSource': {'S3DataSource': {'S3DataType': s3_input_data_type, 'S3Uri': s3_input_uri}}, 'ContentType': content_type, 'CompressionType': compression_type, 'SplitType': split_type}
    transform_output = {'S3OutputPath': s3_output_path, 'Accept': accept, 'AssembleWith': assemble_with}
    transform_resources = {'InstanceType': instance_type, 'InstanceCount': instance_count}
    data_processing = {'InputFilter': input_filter, 'OutputFilter': output_filter, 'JoinSource': join_resource}
    transform_job_response = sage_client.create_transform_job(TransformJobName=job_name, ModelName=model_name, TransformInput=transform_input, TransformOutput=transform_output, TransformResources=transform_resources, DataProcessing=data_processing, Tags=[{'Key': 'model_name', 'Value': model_name}])
    _logger.info('Created batch transform job with arn: %s', transform_job_response['TransformJobArn'])

    def status_check_fn():
        transform_job_info = sage_client.describe_transform_job(TransformJobName=job_name)
        if transform_job_info is None:
            return _SageMakerOperationStatus.in_progress('Waiting for batch transform job to be created...')
        transform_job_status = transform_job_info['TransformJobStatus']
        if transform_job_status == 'InProgress':
            return _SageMakerOperationStatus.in_progress(f'Waiting for batch transform job to reach the "Completed" state.                     Current batch transform job status: "{transform_job_status}"')
        elif transform_job_status == 'Completed':
            return _SageMakerOperationStatus.succeeded('The SageMaker batch transform job was processed successfully.')
        else:
            failure_reason = transform_job_info.get('FailureReason', 'An unknown SageMaker failure occurred. Please see the SageMaker console logs for more information.')
            return _SageMakerOperationStatus.failed(failure_reason)

    def cleanup_fn():
        _logger.info('Cleaning up Sagemaker model and S3 model artifacts...')
        transform_job_info = sage_client.describe_transform_job(TransformJobName=job_name)
        model_name = transform_job_info['ModelName']
        model_arn = _delete_sagemaker_model(model_name, sage_client, s3_client)
        _logger.info('Deleted associated model with arn: %s', model_arn)
    return _SageMakerOperation(status_check_fn=status_check_fn, cleanup_fn=cleanup_fn)