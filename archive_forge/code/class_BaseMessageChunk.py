from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
from langchain_core.load.serializable import Serializable
from langchain_core.pydantic_v1 import Extra, Field
from langchain_core.utils import get_bolded_text
from langchain_core.utils._merge import merge_dicts
from langchain_core.utils.interactive_env import is_interactive_env
class BaseMessageChunk(BaseMessage):
    """Message chunk, which can be concatenated with other Message chunks."""

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'messages']

    def __add__(self, other: Any) -> BaseMessageChunk:
        if isinstance(other, BaseMessageChunk):
            return self.__class__(id=self.id, content=merge_content(self.content, other.content), additional_kwargs=merge_dicts(self.additional_kwargs, other.additional_kwargs), response_metadata=merge_dicts(self.response_metadata, other.response_metadata))
        else:
            raise TypeError(f'unsupported operand type(s) for +: "{self.__class__.__name__}" and "{other.__class__.__name__}"')