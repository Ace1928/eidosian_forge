from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.streamlit.mutable_expander import MutableExpander
def _require_current_thought(self) -> LLMThought:
    """Return our current LLMThought. Raise an error if we have no current
        thought.
        """
    if self._current_thought is None:
        raise RuntimeError('Current LLMThought is unexpectedly None!')
    return self._current_thought