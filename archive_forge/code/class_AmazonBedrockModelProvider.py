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
class AmazonBedrockModelProvider(Enum):
    AMAZON = 'amazon'
    COHERE = 'cohere'
    AI21 = 'ai21'
    ANTHROPIC = 'anthropic'

    @property
    def adapter(self):
        return AWS_MODEL_PROVIDER_TO_ADAPTER.get(self)

    @classmethod
    def of_str(cls, name: str):
        name = name.lower()
        for opt in cls:
            if opt.name.lower() == name or opt.value.lower() == name:
                return opt