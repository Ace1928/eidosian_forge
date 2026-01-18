import json
from typing import Any, Dict, List, Literal, Optional, Tuple
from typing_extensions import TypedDict
from langchain_core.messages.base import (
from langchain_core.utils._merge import merge_dicts
class ToolMessageChunk(ToolMessage, BaseMessageChunk):
    """Tool Message chunk."""
    type: Literal['ToolMessageChunk'] = 'ToolMessageChunk'

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'messages']

    def __add__(self, other: Any) -> BaseMessageChunk:
        if isinstance(other, ToolMessageChunk):
            if self.tool_call_id != other.tool_call_id:
                raise ValueError('Cannot concatenate ToolMessageChunks with different names.')
            return self.__class__(tool_call_id=self.tool_call_id, content=merge_content(self.content, other.content), additional_kwargs=merge_dicts(self.additional_kwargs, other.additional_kwargs), response_metadata=merge_dicts(self.response_metadata, other.response_metadata), id=self.id)
        return super().__add__(other)