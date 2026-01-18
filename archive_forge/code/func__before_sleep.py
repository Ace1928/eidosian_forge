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
def _before_sleep(retry_state: RetryCallState) -> None:
    _logging(retry_state)
    if run_manager:
        if isinstance(run_manager, AsyncCallbackManagerForLLMRun):
            coro = run_manager.on_retry(retry_state)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(coro)
                else:
                    asyncio.run(coro)
            except Exception as e:
                _log_error_once(f'Error in on_retry: {e}')
        else:
            run_manager.on_retry(retry_state)
    return None