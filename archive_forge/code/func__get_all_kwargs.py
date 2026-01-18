from __future__ import annotations
import logging
from typing import (
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import get_from_dict_or_env
def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
    all_kwargs = {**self._default_params, **kwargs}
    for key in list(self._default_params.keys()):
        if all_kwargs.get(key) is None or all_kwargs.get(key) == '':
            all_kwargs.pop(key, None)
    return all_kwargs