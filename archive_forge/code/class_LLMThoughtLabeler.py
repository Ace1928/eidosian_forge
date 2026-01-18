from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.streamlit.mutable_expander import MutableExpander
class LLMThoughtLabeler:
    """
    Generates markdown labels for LLMThought containers. Pass a custom
    subclass of this to StreamlitCallbackHandler to override its default
    labeling logic.
    """

    def get_initial_label(self) -> str:
        """Return the markdown label for a new LLMThought that doesn't have
        an associated tool yet.
        """
        return f'{THINKING_EMOJI} **Thinking...**'

    def get_tool_label(self, tool: ToolRecord, is_complete: bool) -> str:
        """Return the label for an LLMThought that has an associated
        tool.

        Parameters
        ----------
        tool
            The tool's ToolRecord

        is_complete
            True if the thought is complete; False if the thought
            is still receiving input.

        Returns
        -------
        The markdown label for the thought's container.

        """
        input = tool.input_str
        name = tool.name
        emoji = CHECKMARK_EMOJI if is_complete else THINKING_EMOJI
        if name == '_Exception':
            emoji = EXCEPTION_EMOJI
            name = 'Parsing error'
        idx = min([60, len(input)])
        input = input[0:idx]
        if len(tool.input_str) > idx:
            input = input + '...'
        input = input.replace('\n', ' ')
        label = f'{emoji} **{name}:** {input}'
        return label

    def get_history_label(self) -> str:
        """Return a markdown label for the special 'history' container
        that contains overflow thoughts.
        """
        return f'{HISTORY_EMOJI} **History**'

    def get_final_agent_thought_label(self) -> str:
        """Return the markdown label for the agent's final thought -
        the "Now I have the answer" thought, that doesn't involve
        a tool.
        """
        return f'{CHECKMARK_EMOJI} **Complete!**'