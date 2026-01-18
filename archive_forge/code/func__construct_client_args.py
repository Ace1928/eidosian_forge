import json
import time
from enum import Enum
import boto3
import botocore.config
import botocore.exceptions
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from mlflow.gateway.config import AmazonBedrockConfig, AWSIdAndKey, AWSRole, RouteConfig
from mlflow.gateway.constants import (
from mlflow.gateway.exceptions import AIGatewayConfigException
from mlflow.gateway.providers.anthropic import AnthropicAdapter
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.cohere import CohereAdapter
from mlflow.gateway.providers.utils import rename_payload_keys
from mlflow.gateway.schemas import completions
def _construct_client_args(self, session):
    aws_config = self.bedrock_config.aws_config
    if isinstance(aws_config, AWSRole):
        role = session.client(service_name='sts').assume_role(RoleArn=aws_config.aws_role_arn, RoleSessionName='ai-gateway-bedrock', DurationSeconds=aws_config.session_length_seconds)
        return {'aws_access_key_id': role['Credentials']['AccessKeyId'], 'aws_secret_access_key': role['Credentials']['SecretAccessKey'], 'aws_session_token': role['Credentials']['SessionToken']}
    elif isinstance(aws_config, AWSIdAndKey):
        return {'aws_access_key_id': aws_config.aws_access_key_id, 'aws_secret_access_key': aws_config.aws_secret_access_key, 'aws_session_token': aws_config.aws_session_token}
    else:
        return {}