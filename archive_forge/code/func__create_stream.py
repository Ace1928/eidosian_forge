import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional, Union
import aiohttp
import requests
from langchain_core.callbacks import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Extra
def _create_stream(self, api_url: str, payload: Any, stop: Optional[List[str]]=None, **kwargs: Any) -> Iterator[str]:
    if self.stop is not None and stop is not None:
        raise ValueError('`stop` found in both the input and default params.')
    elif self.stop is not None:
        stop = self.stop
    params = self._default_params
    for key in self._default_params:
        if key in kwargs:
            params[key] = kwargs[key]
    if 'options' in kwargs:
        params['options'] = kwargs['options']
    else:
        params['options'] = {**params['options'], 'stop': stop, **{k: v for k, v in kwargs.items() if k not in self._default_params}}
    if payload.get('messages'):
        request_payload = {'messages': payload.get('messages', []), **params}
    else:
        request_payload = {'prompt': payload.get('prompt'), 'images': payload.get('images', []), **params}
    response = requests.post(url=api_url, headers={'Content-Type': 'application/json', **(self.headers if isinstance(self.headers, dict) else {})}, json=request_payload, stream=True, timeout=self.timeout)
    response.encoding = 'utf-8'
    if response.status_code != 200:
        if response.status_code == 404:
            raise OllamaEndpointNotFoundError(f'Ollama call failed with status code 404. Maybe your model is not found and you should pull the model with `ollama pull {self.model}`.')
        else:
            optional_detail = response.text
            raise ValueError(f'Ollama call failed with status code {response.status_code}. Details: {optional_detail}')
    return response.iter_lines(decode_unicode=True)