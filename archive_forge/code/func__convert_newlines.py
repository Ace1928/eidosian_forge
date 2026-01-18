from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.streamlit.mutable_expander import MutableExpander
def _convert_newlines(text: str) -> str:
    """Convert newline characters to markdown newline sequences
    (space, space, newline).
    """
    return text.replace('\n', '  \n')