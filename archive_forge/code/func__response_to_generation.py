from __future__ import annotations
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterator, List, Optional, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks.manager import (
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_community.utilities.vertexai import (
def _response_to_generation(self, response: TextGenerationResponse) -> GenerationChunk:
    """Converts a stream response to a generation chunk."""
    try:
        generation_info = {'is_blocked': response.is_blocked, 'safety_attributes': response.safety_attributes}
    except Exception:
        generation_info = None
    return GenerationChunk(text=response.text, generation_info=generation_info)