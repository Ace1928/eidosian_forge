from __future__ import annotations
import logging
import os
import warnings
from typing import (
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env, get_pydantic_field_names
from tenacity import (
from langchain_community.utils.openai import is_openai_v1
def _async_retry_decorator(embeddings: OpenAIEmbeddings) -> Any:
    import openai
    async_retrying = AsyncRetrying(reraise=True, stop=stop_after_attempt(embeddings.max_retries), wait=wait_exponential(multiplier=1, min=embeddings.retry_min_seconds, max=embeddings.retry_max_seconds), retry=retry_if_exception_type(openai.error.Timeout) | retry_if_exception_type(openai.error.APIError) | retry_if_exception_type(openai.error.APIConnectionError) | retry_if_exception_type(openai.error.RateLimitError) | retry_if_exception_type(openai.error.ServiceUnavailableError), before_sleep=before_sleep_log(logger, logging.WARNING))

    def wrap(func: Callable) -> Callable:

        async def wrapped_f(*args: Any, **kwargs: Any) -> Callable:
            async for _ in async_retrying:
                return await func(*args, **kwargs)
            raise AssertionError('this is unreachable')
        return wrapped_f
    return wrap