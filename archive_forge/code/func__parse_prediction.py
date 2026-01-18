from __future__ import annotations
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterator, List, Optional, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks.manager import (
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_community.utilities.vertexai import (
def _parse_prediction(self, prediction: Any) -> str:
    if isinstance(prediction, str):
        return prediction
    if self.result_arg:
        try:
            return prediction[self.result_arg]
        except KeyError:
            if isinstance(prediction, str):
                error_desc = f'Provided non-None `result_arg` (result_arg={self.result_arg}). But got prediction of type {type(prediction)} instead of dict. Most probably, youneed to set `result_arg=None` during VertexAIModelGarden initialization.'
                raise ValueError(error_desc)
            else:
                raise ValueError(f'{self.result_arg} key not found in prediction!')
    return prediction