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
class UserContextManager:
    """Context manager for LLMonitor user context."""

    def __init__(self, user_id: str, user_props: Any=None) -> None:
        user_ctx.set(user_id)
        user_props_ctx.set(user_props)

    def __enter__(self) -> Any:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> Any:
        user_ctx.set(None)
        user_props_ctx.set(None)