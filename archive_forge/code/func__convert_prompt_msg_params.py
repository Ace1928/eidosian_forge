from __future__ import annotations
import logging
from typing import (
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
def _convert_prompt_msg_params(self, prompt: str, **kwargs: Any) -> dict:
    if 'streaming' in kwargs:
        kwargs['stream'] = kwargs.pop('streaming')
    return {**{'prompt': prompt, 'model': self.model}, **self._default_params, **kwargs}