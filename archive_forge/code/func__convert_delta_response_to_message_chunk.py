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
def _convert_delta_response_to_message_chunk(response: ChatCompletionResponseStream, default_class: Type[BaseMessageChunk]) -> Tuple[Union[BaseMessageChunk, HumanMessageChunk, AIMessageChunk, SystemMessageChunk], Optional[str]]:
    """Converts delta response to message chunk"""
    _delta = response.choices[0].delta
    role = _delta.get('role', '')
    content = _delta.get('content', '')
    additional_kwargs: Dict = {}
    if role is None or role == '':
        raise ChatPremAPIError('Role can not be None. Please check the response')
    finish_reasons: Optional[str] = response.choices[0].finish_reason
    if role == 'user' or default_class == HumanMessageChunk:
        return (HumanMessageChunk(content=content), finish_reasons)
    elif role == 'assistant' or default_class == AIMessageChunk:
        return (AIMessageChunk(content=content, additional_kwargs=additional_kwargs), finish_reasons)
    elif role == 'system' or default_class == SystemMessageChunk:
        return (SystemMessageChunk(content=content), finish_reasons)
    elif role or default_class == ChatMessageChunk:
        return (ChatMessageChunk(content=content, role=role), finish_reasons)
    else:
        return (default_class(content=content), finish_reasons)