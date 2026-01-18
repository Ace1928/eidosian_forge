from __future__ import annotations
import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core.callbacks.manager import (
from langchain_core.language_models.llms import LLM
from langchain_core.load.serializable import Serializable
from langchain_core.outputs import GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils.env import get_from_dict_or_env
from langchain_core.utils.utils import convert_to_secret_str
def _get_invocation_params(self, stop: Optional[List[str]]=None, **kwargs: Any) -> Dict[str, Any]:
    """Get the parameters used to invoke the model."""
    params = self._default_params
    if self.stop is not None and stop is not None:
        raise ValueError('`stop` found in both the input and default params.')
    elif self.stop is not None:
        params['stop'] = self.stop
    else:
        params['stop'] = stop
    return {**params, **kwargs}