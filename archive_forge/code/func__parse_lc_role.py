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
def _parse_lc_role(role: str) -> str:
    if role == 'human':
        return 'user'
    else:
        return role