from __future__ import annotations
import asyncio
import functools
import inspect
import json
import logging
import uuid
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
import yaml
from tenacity import (
from langchain_core._api import deprecated
from langchain_core.caches import BaseCache
from langchain_core.callbacks import (
from langchain_core.globals import get_llm_cache
from langchain_core.language_models.base import BaseLanguageModel, LanguageModelInput
from langchain_core.load import dumpd
from langchain_core.messages import (
from langchain_core.outputs import Generation, GenerationChunk, LLMResult, RunInfo
from langchain_core.prompt_values import ChatPromptValue, PromptValue, StringPromptValue
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.runnables import RunnableConfig, ensure_config, get_config_list
from langchain_core.runnables.config import run_in_executor
def _resolve_cache(cache: Union[BaseCache, bool, None]) -> Optional[BaseCache]:
    """Resolve the cache."""
    if isinstance(cache, BaseCache):
        llm_cache = cache
    elif cache is None:
        llm_cache = get_llm_cache()
    elif cache is True:
        llm_cache = get_llm_cache()
        if llm_cache is None:
            raise ValueError('No global cache was configured. Use `set_llm_cache`.to set a global cache if you want to use a global cache.Otherwise either pass a cache object or set cache to False/None')
    elif cache is False:
        llm_cache = None
    else:
        raise ValueError(f'Unsupported cache value {cache}')
    return llm_cache