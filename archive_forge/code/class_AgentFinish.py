from __future__ import annotations
import json
from typing import Any, List, Literal, Sequence, Union
from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
class AgentFinish(Serializable):
    """The final return value of an ActionAgent."""
    return_values: dict
    'Dictionary of return values.'
    log: str
    'Additional information to log about the return value.\n    This is used to pass along the full LLM prediction, not just the parsed out\n    return value. For example, if the full LLM prediction was\n    `Final Answer: 2` you may want to just return `2` as a return value, but pass\n    along the full string as a `log` (for debugging or observability purposes).\n    '
    type: Literal['AgentFinish'] = 'AgentFinish'

    def __init__(self, return_values: dict, log: str, **kwargs: Any):
        """Override init to support instantiation by position for backward compat."""
        super().__init__(return_values=return_values, log=log, **kwargs)

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
        """Return the messages that correspond to this observation."""
        return [AIMessage(content=self.log)]