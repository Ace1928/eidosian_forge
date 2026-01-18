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
class ContentFormatterBase:
    """Transform request and response of AzureML endpoint to match with
    required schema.
    """
    '\n    Example:\n        .. code-block:: python\n        \n            class ContentFormatter(ContentFormatterBase):\n                content_type = "application/json"\n                accepts = "application/json"\n                \n                def format_request_payload(\n                    self, \n                    prompt: str, \n                    model_kwargs: Dict,\n                    api_type: AzureMLEndpointApiType,\n                ) -> bytes:\n                    input_str = json.dumps(\n                        {\n                            "inputs": {"input_string": [prompt]}, \n                            "parameters": model_kwargs,\n                        }\n                    )\n                    return str.encode(input_str)\n                    \n                def format_response_payload(\n                        self, output: str, api_type: AzureMLEndpointApiType\n                    ) -> str:\n                    response_json = json.loads(output)\n                    return response_json[0]["0"]\n    '
    content_type: Optional[str] = 'application/json'
    'The MIME type of the input data passed to the endpoint'
    accepts: Optional[str] = 'application/json'
    'The MIME type of the response data returned from the endpoint'
    format_error_msg: str = 'Error while formatting response payload for chat model of type  `{api_type}`. Are you using the right formatter for the deployed  model and endpoint type?'

    @staticmethod
    def escape_special_characters(prompt: str) -> str:
        """Escapes any special characters in `prompt`"""
        escape_map = {'\\': '\\\\', '"': '\\"', '\x08': '\\b', '\x0c': '\\f', '\n': '\\n', '\r': '\\r', '\t': '\\t'}
        for escape_sequence, escaped_sequence in escape_map.items():
            prompt = prompt.replace(escape_sequence, escaped_sequence)
        return prompt

    @property
    def supported_api_types(self) -> List[AzureMLEndpointApiType]:
        """Supported APIs for the given formatter. Azure ML supports
        deploying models using different hosting methods. Each method may have
        a different API structure."""
        return [AzureMLEndpointApiType.dedicated]

    def format_request_payload(self, prompt: str, model_kwargs: Dict, api_type: AzureMLEndpointApiType=AzureMLEndpointApiType.dedicated) -> Any:
        """Formats the request body according to the input schema of
        the model. Returns bytes or seekable file like object in the
        format specified in the content_type request header.
        """
        raise NotImplementedError()

    @abstractmethod
    def format_response_payload(self, output: bytes, api_type: AzureMLEndpointApiType=AzureMLEndpointApiType.dedicated) -> Generation:
        """Formats the response body according to the output
        schema of the model. Returns the data type that is
        received from the response.
        """