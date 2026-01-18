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
class AmazonBedrockProvider(BaseProvider):
    NAME = 'Amazon Bedrock'

    def __init__(self, config: RouteConfig):
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, AmazonBedrockConfig):
            raise TypeError(f'Invalid config type {config.model.config}')
        self.bedrock_config: AmazonBedrockConfig = config.model.config
        self._client = None
        self._client_created = 0

    def _client_expired(self):
        if not isinstance(self.bedrock_config.aws_config, AWSRole):
            return False
        return (time.monotonic_ns() - self._client_created >= self.bedrock_config.aws_config.session_length_seconds * 1000000000,)

    def get_bedrock_client(self):
        if self._client is not None and (not self._client_expired()):
            return self._client
        session = boto3.Session(**self._construct_session_args())
        try:
            self._client, self._client_created = (session.client(service_name='bedrock-runtime', **self._construct_client_args(session)), time.monotonic_ns())
            return self._client
        except botocore.exceptions.UnknownServiceError as e:
            raise AIGatewayConfigException('Cannot create Amazon Bedrock client; ensure boto3/botocore linked from the Amazon Bedrock user guide are installed. Otherwise likely missing credentials or accessing account without to Amazon Bedrock Private Preview') from e

    def _construct_session_args(self):
        session_args = {'region_name': self.bedrock_config.aws_config.aws_region}
        return {k: v for k, v in session_args.items() if v}

    def _construct_client_args(self, session):
        aws_config = self.bedrock_config.aws_config
        if isinstance(aws_config, AWSRole):
            role = session.client(service_name='sts').assume_role(RoleArn=aws_config.aws_role_arn, RoleSessionName='ai-gateway-bedrock', DurationSeconds=aws_config.session_length_seconds)
            return {'aws_access_key_id': role['Credentials']['AccessKeyId'], 'aws_secret_access_key': role['Credentials']['SecretAccessKey'], 'aws_session_token': role['Credentials']['SessionToken']}
        elif isinstance(aws_config, AWSIdAndKey):
            return {'aws_access_key_id': aws_config.aws_access_key_id, 'aws_secret_access_key': aws_config.aws_secret_access_key, 'aws_session_token': aws_config.aws_session_token}
        else:
            return {}

    @property
    def _underlying_provider(self):
        if not self.config.model.name or '.' not in self.config.model.name:
            return None
        provider = self.config.model.name.split('.')[0]
        return AmazonBedrockModelProvider.of_str(provider)

    @property
    def underlying_provider_adapter(self) -> ProviderAdapter:
        provider = self._underlying_provider
        if not provider:
            raise HTTPException(status_code=422, detail=f'Unknown Amazon Bedrock model type {self._underlying_provider}')
        adapter = provider.adapter
        if not adapter:
            raise HTTPException(status_code=422, detail=f"Don't know how to handle {self._underlying_provider} for Amazon Bedrock")
        return adapter

    def _request(self, body):
        try:
            response = self.get_bedrock_client().invoke_model(body=json.dumps(body).encode(), modelId=self.config.model.name, accept='application/json', contentType='application/json')
            return json.loads(response.get('body').read())
        except botocore.exceptions.ReadTimeoutError as e:
            raise HTTPException(status_code=408) from e

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        self.check_for_model_field(payload)
        payload = jsonable_encoder(payload, exclude_none=True, exclude_defaults=True)
        payload = self.underlying_provider_adapter.completions_to_model(payload, self.config)
        response = self._request(payload)
        return self.underlying_provider_adapter.model_to_completions(response, self.config)