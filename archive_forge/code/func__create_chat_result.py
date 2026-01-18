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
def _create_chat_result(self, response: Union[dict, BaseModel]) -> ChatResult:
    generations = []
    if not isinstance(response, dict):
        response = response.dict()
    for res in response['choices']:
        message = convert_dict_to_message(res['message'])
        generation_info = dict(finish_reason=res.get('finish_reason'))
        if 'logprobs' in res:
            generation_info['logprobs'] = res['logprobs']
        gen = ChatGeneration(message=message, generation_info=generation_info)
        generations.append(gen)
    token_usage = response.get('usage', {})
    llm_output = {'token_usage': token_usage, 'model_name': self.model_name, 'system_fingerprint': response.get('system_fingerprint', '')}
    return ChatResult(generations=generations, llm_output=llm_output)