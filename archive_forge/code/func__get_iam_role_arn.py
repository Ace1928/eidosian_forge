import json
import os
from ...utils.constants import SAGEMAKER_PARALLEL_EC2_INSTANCES, TORCH_DYNAMO_MODES
from ...utils.dataclasses import ComputeEnvironment, SageMakerDistributedType
from ...utils.imports import is_boto3_available
from .config_args import SageMakerConfig
from .config_utils import (
def _get_iam_role_arn(role_name):
    iam_client = boto3.client('iam')
    return iam_client.get_role(RoleName=role_name)['Role']['Arn']