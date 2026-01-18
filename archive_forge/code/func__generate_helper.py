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
def _generate_helper(self, prompts: List[str], stop: Optional[List[str]], run_managers: List[CallbackManagerForLLMRun], new_arg_supported: bool, **kwargs: Any) -> LLMResult:
    try:
        output = self._generate(prompts, stop=stop, run_manager=run_managers[0] if run_managers else None, **kwargs) if new_arg_supported else self._generate(prompts, stop=stop)
    except BaseException as e:
        for run_manager in run_managers:
            run_manager.on_llm_error(e, response=LLMResult(generations=[]))
        raise e
    flattened_outputs = output.flatten()
    for manager, flattened_output in zip(run_managers, flattened_outputs):
        manager.on_llm_end(flattened_output)
    if run_managers:
        output.run = [RunInfo(run_id=run_manager.run_id) for run_manager in run_managers]
    return output