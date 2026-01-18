from __future__ import annotations
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterator, List, Optional, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks.manager import (
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_community.utilities.vertexai import (
@classmethod
def _get_task_executor(cls, request_parallelism: int=5) -> Executor:
    if cls.task_executor is None:
        cls.task_executor = ThreadPoolExecutor(max_workers=request_parallelism)
    return cls.task_executor