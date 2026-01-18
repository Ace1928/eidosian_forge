from __future__ import annotations
import json
from typing import Any, List, Literal, Sequence, Union
from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
class AgentStep(Serializable):
    """The result of running an AgentAction."""
    action: AgentAction
    'The AgentAction that was executed.'
    observation: Any
    'The result of the AgentAction.'

    @property
    def messages(self) -> Sequence[BaseMessage]:
        """Return the messages that correspond to this observation."""
        return _convert_agent_observation_to_messages(self.action, self.observation)