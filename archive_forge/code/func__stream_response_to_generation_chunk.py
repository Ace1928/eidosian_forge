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
def _stream_response_to_generation_chunk(stream_response: Dict[str, Any]) -> GenerationChunk:
    """Convert a stream response to a generation chunk."""
    if not stream_response['choices']:
        return GenerationChunk(text='')
    return GenerationChunk(text=stream_response['choices'][0]['text'], generation_info=dict(finish_reason=stream_response['choices'][0].get('finish_reason', None), logprobs=stream_response['choices'][0].get('logprobs', None)))