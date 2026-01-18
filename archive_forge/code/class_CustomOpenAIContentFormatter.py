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
class CustomOpenAIContentFormatter(ContentFormatterBase):
    """Content formatter for models that use the OpenAI like API scheme."""

    @property
    def supported_api_types(self) -> List[AzureMLEndpointApiType]:
        return [AzureMLEndpointApiType.dedicated, AzureMLEndpointApiType.serverless]

    def format_request_payload(self, prompt: str, model_kwargs: Dict, api_type: AzureMLEndpointApiType) -> bytes:
        """Formats the request according to the chosen api"""
        prompt = ContentFormatterBase.escape_special_characters(prompt)
        if api_type in [AzureMLEndpointApiType.dedicated, AzureMLEndpointApiType.realtime]:
            request_payload = json.dumps({'input_data': {'input_string': [f'"{prompt}"'], 'parameters': model_kwargs}})
        elif api_type == AzureMLEndpointApiType.serverless:
            request_payload = json.dumps({'prompt': prompt, **model_kwargs})
        else:
            raise ValueError(f'`api_type` {api_type} is not supported by this formatter')
        return str.encode(request_payload)

    def format_response_payload(self, output: bytes, api_type: AzureMLEndpointApiType) -> Generation:
        """Formats response"""
        if api_type in [AzureMLEndpointApiType.dedicated, AzureMLEndpointApiType.realtime]:
            try:
                choice = json.loads(output)[0]['0']
            except (KeyError, IndexError, TypeError) as e:
                raise ValueError(self.format_error_msg.format(api_type=api_type)) from e
            return Generation(text=choice)
        if api_type == AzureMLEndpointApiType.serverless:
            try:
                choice = json.loads(output)['choices'][0]
                if not isinstance(choice, dict):
                    raise TypeError('Endpoint response is not well formed for a chat model. Expected `dict` but `{type(choice)}` was received.')
            except (KeyError, IndexError, TypeError) as e:
                raise ValueError(self.format_error_msg.format(api_type=api_type)) from e
            return Generation(text=choice['text'].strip(), generation_info=dict(finish_reason=choice.get('finish_reason'), logprobs=choice.get('logprobs')))
        raise ValueError(f'`api_type` {api_type} is not supported by this formatter')