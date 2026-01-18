from __future__ import annotations
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterator, List, Optional, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks.manager import (
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_community.utilities.vertexai import (
class _VertexAIBase(BaseModel):
    project: Optional[str] = None
    'The default GCP project to use when making Vertex API calls.'
    location: str = 'us-central1'
    'The default location to use when making API calls.'
    request_parallelism: int = 5
    'The amount of parallelism allowed for requests issued to VertexAI models. '
    'Default is 5.'
    max_retries: int = 6
    'The maximum number of retries to make when generating.'
    task_executor: ClassVar[Optional[Executor]] = Field(default=None, exclude=True)
    stop: Optional[List[str]] = None
    'Optional list of stop words to use when generating.'
    model_name: Optional[str] = None
    'Underlying model name.'

    @classmethod
    def _get_task_executor(cls, request_parallelism: int=5) -> Executor:
        if cls.task_executor is None:
            cls.task_executor = ThreadPoolExecutor(max_workers=request_parallelism)
        return cls.task_executor