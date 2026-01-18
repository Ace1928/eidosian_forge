from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Literal, Sequence, cast
from typing_extensions import TypedDict
from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
class ChatPromptValueConcrete(ChatPromptValue):
    """Chat prompt value which explicitly lists out the message types it accepts.
    For use in external schemas."""
    messages: Sequence[AnyMessage]
    type: Literal['ChatPromptValueConcrete'] = 'ChatPromptValueConcrete'

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'prompts', 'chat']