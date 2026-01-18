from __future__ import annotations
import logging
import os
import sys
from typing import (
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.runnables import Runnable
from langchain_core.utils import (
from langchain_community.adapters.openai import (
from langchain_community.utils.openai import is_openai_v1
def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
    overall_token_usage: dict = {}
    system_fingerprint = None
    for output in llm_outputs:
        if output is None:
            continue
        token_usage = output['token_usage']
        if token_usage is not None:
            for k, v in token_usage.items():
                if k in overall_token_usage:
                    overall_token_usage[k] += v
                else:
                    overall_token_usage[k] = v
        if system_fingerprint is None:
            system_fingerprint = output.get('system_fingerprint')
    combined = {'token_usage': overall_token_usage, 'model_name': self.model_name}
    if system_fingerprint:
        combined['system_fingerprint'] = system_fingerprint
    return combined