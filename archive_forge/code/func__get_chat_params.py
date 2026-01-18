from __future__ import annotations
import logging
import os
import sys
import warnings
from typing import (
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.llms import BaseLLM, create_base_retry_decorator
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env, get_pydantic_field_names
from langchain_core.utils.utils import build_extra_kwargs
from langchain_community.utils.openai import is_openai_v1
def _get_chat_params(self, prompts: List[str], stop: Optional[List[str]]=None) -> Tuple:
    if len(prompts) > 1:
        raise ValueError(f'OpenAIChat currently only supports single prompt, got {prompts}')
    messages = self.prefix_messages + [{'role': 'user', 'content': prompts[0]}]
    params: Dict[str, Any] = {**{'model': self.model_name}, **self._default_params}
    if stop is not None:
        if 'stop' in params:
            raise ValueError('`stop` found in both the input and default params.')
        params['stop'] = stop
    if params.get('max_tokens') == -1:
        del params['max_tokens']
    return (messages, params)