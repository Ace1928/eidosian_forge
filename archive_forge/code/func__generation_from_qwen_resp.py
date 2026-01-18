from __future__ import annotations
import asyncio
import functools
import logging
from typing import (
from langchain_core.callbacks import (
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env
from requests.exceptions import HTTPError
from tenacity import (
@staticmethod
def _generation_from_qwen_resp(resp: Any, is_last_chunk: bool=True) -> Dict[str, Any]:
    if is_last_chunk:
        return dict(text=resp['output']['text'], generation_info=dict(finish_reason=resp['output']['finish_reason'], request_id=resp['request_id'], token_usage=dict(resp['usage'])))
    else:
        return dict(text=resp['output']['text'])