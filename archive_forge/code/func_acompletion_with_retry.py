from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_core.load.serializable import Serializable
from langchain_core.pydantic_v1 import Extra, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from tenacity import (
from langchain_community.llms.utils import enforce_stop_tokens
def acompletion_with_retry(llm: Cohere, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm.max_retries)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        return await llm.async_client.generate(**kwargs)
    return _completion_with_retry(**kwargs)