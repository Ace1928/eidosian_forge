from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Literal, Sequence, cast
from typing_extensions import TypedDict
from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
class StringPromptValue(PromptValue):
    """String prompt value."""
    text: str
    'Prompt text.'
    type: Literal['StringPromptValue'] = 'StringPromptValue'

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'prompts', 'base']

    def to_string(self) -> str:
        """Return prompt as string."""
        return self.text

    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as messages."""
        return [HumanMessage(content=self.text)]