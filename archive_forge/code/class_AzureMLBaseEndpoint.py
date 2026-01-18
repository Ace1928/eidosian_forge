import json
import urllib.request
import warnings
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator, validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
class AzureMLBaseEndpoint(BaseModel):
    """Azure ML Online Endpoint models."""
    endpoint_url: str = ''
    'URL of pre-existing Endpoint. Should be passed to constructor or specified as \n        env var `AZUREML_ENDPOINT_URL`.'
    endpoint_api_type: AzureMLEndpointApiType = AzureMLEndpointApiType.dedicated
    'Type of the endpoint being consumed. Possible values are `serverless` for \n        pay-as-you-go and `dedicated` for dedicated endpoints. '
    endpoint_api_key: SecretStr = convert_to_secret_str('')
    'Authentication Key for Endpoint. Should be passed to constructor or specified as\n        env var `AZUREML_ENDPOINT_API_KEY`.'
    deployment_name: str = ''
    'Deployment Name for Endpoint. NOT REQUIRED to call endpoint. Should be passed \n        to constructor or specified as env var `AZUREML_DEPLOYMENT_NAME`.'
    timeout: int = DEFAULT_TIMEOUT
    'Request timeout for calls to the endpoint'
    http_client: Any = None
    max_retries: int = 1
    content_formatter: Any = None
    'The content formatter that provides an input and output\n    transform function to handle formats between the LLM and\n    the endpoint'
    model_kwargs: Optional[dict] = None
    'Keyword arguments to pass to the model.'

    @root_validator(pre=True)
    def validate_environ(cls, values: Dict) -> Dict:
        values['endpoint_api_key'] = convert_to_secret_str(get_from_dict_or_env(values, 'endpoint_api_key', 'AZUREML_ENDPOINT_API_KEY'))
        values['endpoint_url'] = get_from_dict_or_env(values, 'endpoint_url', 'AZUREML_ENDPOINT_URL')
        values['deployment_name'] = get_from_dict_or_env(values, 'deployment_name', 'AZUREML_DEPLOYMENT_NAME', '')
        values['endpoint_api_type'] = get_from_dict_or_env(values, 'endpoint_api_type', 'AZUREML_ENDPOINT_API_TYPE', AzureMLEndpointApiType.dedicated)
        values['timeout'] = get_from_dict_or_env(values, 'timeout', 'AZUREML_TIMEOUT', str(DEFAULT_TIMEOUT))
        return values

    @validator('content_formatter')
    def validate_content_formatter(cls, field_value: Any, values: Dict) -> ContentFormatterBase:
        """Validate that content formatter is supported by endpoint type."""
        endpoint_api_type = values.get('endpoint_api_type')
        if endpoint_api_type not in field_value.supported_api_types:
            raise ValueError(f'Content formatter f{type(field_value)} is not supported by this endpoint. Supported types are {field_value.supported_api_types} but endpoint is {endpoint_api_type}.')
        return field_value

    @validator('endpoint_url')
    def validate_endpoint_url(cls, field_value: Any) -> str:
        """Validate that endpoint url is complete."""
        if field_value.endswith('/'):
            field_value = field_value[:-1]
        if field_value.endswith('inference.ml.azure.com'):
            raise ValueError("`endpoint_url` should contain the full invocation URL including `/score` for `endpoint_api_type='dedicated'` or `/v1/completions` or `/v1/chat/completions` for `endpoint_api_type='serverless'`")
        return field_value

    @validator('endpoint_api_type')
    def validate_endpoint_api_type(cls, field_value: Any, values: Dict) -> AzureMLEndpointApiType:
        """Validate that endpoint api type is compatible with the URL format."""
        endpoint_url = values.get('endpoint_url')
        if (field_value == AzureMLEndpointApiType.dedicated or field_value == AzureMLEndpointApiType.realtime) and (not endpoint_url.endswith('/score')):
            raise ValueError("Endpoints of type `dedicated` should follow the format `https://<your-endpoint>.<your_region>.inference.ml.azure.com/score`. If your endpoint URL ends with `/v1/completions` or`/v1/chat/completions`, use `endpoint_api_type='serverless'` instead.")
        if field_value == AzureMLEndpointApiType.serverless and (not (endpoint_url.endswith('/v1/completions') or endpoint_url.endswith('/v1/chat/completions'))):
            raise ValueError('Endpoints of type `serverless` should follow the format `https://<your-endpoint>.<your_region>.inference.ml.azure.com/v1/chat/completions` or `https://<your-endpoint>.<your_region>.inference.ml.azure.com/v1/chat/completions`')
        return field_value

    @validator('http_client', always=True)
    def validate_client(cls, field_value: Any, values: Dict) -> AzureMLEndpointClient:
        """Validate that api key and python package exists in environment."""
        endpoint_url = values.get('endpoint_url')
        endpoint_key = values.get('endpoint_api_key')
        deployment_name = values.get('deployment_name')
        timeout = values.get('timeout', DEFAULT_TIMEOUT)
        http_client = AzureMLEndpointClient(endpoint_url, endpoint_key.get_secret_value(), deployment_name, timeout)
        return http_client