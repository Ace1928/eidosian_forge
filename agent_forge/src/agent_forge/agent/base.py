"""
Base Agent Class for Eidosian Forge.
"""

from abc import ABC, abstractmethod
from typing import Any

from agent_forge.models import ThoughtType


class BaseAgent(ABC):
    """
    Abstract base class for an agent in the Eidosian Forge system.
    """

    @abstractmethod
    def get_memory(self) -> Any:
        """
        Get the agent's memory system.
        """
        pass

    @abstractmethod
    def get_model_manager(self) -> Any:
        """
        Get the agent's model manager.
        """
        pass

    @abstractmethod
    def get_sandbox(self) -> Any:
        """
        Get the agent's sandbox.
        """
        pass

    @abstractmethod
    def add_thought(self, content: str, thought_type: ThoughtType) -> str:
        """
        Add a thought to the agent's memory.
        """
        pass
