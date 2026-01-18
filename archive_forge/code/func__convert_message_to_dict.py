import json
import logging
from typing import Any, List, Optional, Union
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.messages import (
from langchain_core.pydantic_v1 import Field
from langchain_community.llms.utils import enforce_stop_tokens
def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, HumanMessage):
        message_dict = {'role': 'user', 'content': message.content}
    elif isinstance(message, AIMessage):
        message_dict = {'role': 'assistant', 'content': message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {'role': 'system', 'content': message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {'role': 'function', 'content': message.content}
    else:
        raise ValueError(f'Got unknown type {message}')
    return message_dict