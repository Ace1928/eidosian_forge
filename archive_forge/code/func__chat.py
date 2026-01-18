import json
import logging
from typing import Any, Dict, Iterator, List, Mapping, Optional, Type
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import (
def _chat(self, messages: List[BaseMessage], **kwargs: Any) -> requests.Response:
    parameters = {**self._default_params, **kwargs}
    model = parameters.pop('model')
    headers = parameters.pop('headers', {})
    temperature = parameters.pop('temperature', 0.3)
    top_k = parameters.pop('top_k', 5)
    top_p = parameters.pop('top_p', 0.85)
    with_search_enhance = parameters.pop('with_search_enhance', False)
    stream = parameters.pop('stream', False)
    payload = {'model': model, 'messages': [_convert_message_to_dict(m) for m in messages], 'top_k': top_k, 'top_p': top_p, 'temperature': temperature, 'with_search_enhance': with_search_enhance, 'stream': stream}
    url = self.baichuan_api_base
    api_key = ''
    if self.baichuan_api_key:
        api_key = self.baichuan_api_key.get_secret_value()
    res = requests.post(url=url, timeout=self.request_timeout, headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}', **headers}, json=payload, stream=self.streaming)
    return res