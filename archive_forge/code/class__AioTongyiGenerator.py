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
class _AioTongyiGenerator:

    def __init__(self, _llm: Tongyi, **_kwargs: Any):
        self.generator = stream_generate_with_retry(_llm, **_kwargs)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        value = await asyncio.get_running_loop().run_in_executor(None, self._safe_next)
        if value is not None:
            return value
        else:
            raise StopAsyncIteration

    def _safe_next(self) -> Any:
        try:
            return next(self.generator)
        except StopIteration:
            return None