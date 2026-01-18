from typing import Any, List, Literal
from langchain_core.messages.base import (
from langchain_core.utils._merge import merge_dicts
class ChatMessageChunk(ChatMessage, BaseMessageChunk):
    """Chat Message chunk."""
    type: Literal['ChatMessageChunk'] = 'ChatMessageChunk'

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'messages']

    def __add__(self, other: Any) -> BaseMessageChunk:
        if isinstance(other, ChatMessageChunk):
            if self.role != other.role:
                raise ValueError('Cannot concatenate ChatMessageChunks with different roles.')
            return self.__class__(role=self.role, content=merge_content(self.content, other.content), additional_kwargs=merge_dicts(self.additional_kwargs, other.additional_kwargs), response_metadata=merge_dicts(self.response_metadata, other.response_metadata), id=self.id)
        elif isinstance(other, BaseMessageChunk):
            return self.__class__(role=self.role, content=merge_content(self.content, other.content), additional_kwargs=merge_dicts(self.additional_kwargs, other.additional_kwargs), response_metadata=merge_dicts(self.response_metadata, other.response_metadata), id=self.id)
        else:
            return super().__add__(other)