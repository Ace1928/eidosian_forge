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
def _try_init_vertexai(cls, values: Dict) -> None:
    allowed_params = ['project', 'location', 'credentials']
    params = {k: v for k, v in values.items() if k in allowed_params}
    init_vertexai(**params)
    return None