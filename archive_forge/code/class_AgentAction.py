from __future__ import annotations
import json
from typing import Any, List, Literal, Sequence, Union
from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
class AgentAction(Serializable):
    """A full description of an action for an ActionAgent to execute."""
    tool: str
    'The name of the Tool to execute.'
    tool_input: Union[str, dict]
    'The input to pass in to the Tool.'
    log: str
    'Additional information to log about the action.\n    This log can be used in a few ways. First, it can be used to audit\n    what exactly the LLM predicted to lead to this (tool, tool_input).\n    Second, it can be used in future iterations to show the LLMs prior\n    thoughts. This is useful when (tool, tool_input) does not contain\n    full information about the LLM prediction (for example, any `thought`\n    before the tool/tool_input).'
    type: Literal['AgentAction'] = 'AgentAction'

    def __init__(self, tool: str, tool_input: Union[str, dict], log: str, **kwargs: Any):
        """Override init to support instantiation by position for backward compat."""
        super().__init__(tool=tool, tool_input=tool_input, log=log, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether or not the class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'agent']

    @property
    def messages(self) -> Sequence[BaseMessage]:
        """Return the messages that correspond to this action."""
        return _convert_agent_action_to_messages(self)