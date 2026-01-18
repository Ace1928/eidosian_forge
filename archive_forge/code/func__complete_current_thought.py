from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.streamlit.mutable_expander import MutableExpander
def _complete_current_thought(self, final_label: Optional[str]=None) -> None:
    """Complete the current thought, optionally assigning it a new label.
        Add it to our _completed_thoughts list.
        """
    thought = self._require_current_thought()
    thought.complete(final_label)
    self._completed_thoughts.append(thought)
    self._current_thought = None