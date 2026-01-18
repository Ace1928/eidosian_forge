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
def _get_user_props(metadata: Any) -> Any:
    if user_props_ctx.get() is not None:
        return user_props_ctx.get()
    metadata = metadata or {}
    return metadata.get('user_props', None)