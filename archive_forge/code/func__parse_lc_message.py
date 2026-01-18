import importlib.metadata
import logging
import os
import traceback
import warnings
from contextvars import ContextVar
from typing import Any, Dict, List, Union, cast
from uuid import UUID
import requests
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from packaging.version import parse
def _parse_lc_message(message: BaseMessage) -> Dict[str, Any]:
    keys = ['function_call', 'tool_calls', 'tool_call_id', 'name']
    parsed = {'text': message.content, 'role': _parse_lc_role(message.type)}
    parsed.update({key: cast(Any, message.additional_kwargs.get(key)) for key in keys if message.additional_kwargs.get(key) is not None})
    return parsed