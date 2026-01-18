from __future__ import annotations
from typing import Any, Dict, Iterator, List, Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.llms import BaseLLM
from langchain_community.utilities.vertexai import create_retry_decorator
def _strip_erroneous_leading_spaces(text: str) -> str:
    """Strip erroneous leading spaces from text.

    The PaLM API will sometimes erroneously return a single leading space in all
    lines > 1. This function strips that space.
    """
    has_leading_space = all((not line or line[0] == ' ' for line in text.split('\n')[1:]))
    if has_leading_space:
        return text.replace('\n ', '\n')
    else:
        return text