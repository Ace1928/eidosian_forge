import json
from typing import AsyncIterable
from mlflow.exceptions import MlflowException
from mlflow.gateway.config import OpenAIAPIType, OpenAIConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import send_request, send_stream_request
from mlflow.gateway.schemas import chat, completions, embeddings
from mlflow.gateway.utils import handle_incomplete_chunks, strip_sse_prefix
from mlflow.utils.uri import append_to_uri_path, append_to_uri_query_params
@property
def _request_headers(self):
    api_type = self.openai_config.openai_api_type
    if api_type == OpenAIAPIType.OPENAI:
        headers = {'Authorization': f'Bearer {self.openai_config.openai_api_key}'}
        if (org := self.openai_config.openai_organization):
            headers['OpenAI-Organization'] = org
        return headers
    elif api_type == OpenAIAPIType.AZUREAD:
        return {'Authorization': f'Bearer {self.openai_config.openai_api_key}'}
    elif api_type == OpenAIAPIType.AZURE:
        return {'api-key': self.openai_config.openai_api_key}
    else:
        raise MlflowException.invalid_parameter_value(f"Invalid OpenAI API type '{self.openai_config.openai_api_type}'")