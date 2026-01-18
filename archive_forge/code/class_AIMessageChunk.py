from typing import Any, Dict, List, Literal
from langchain_core.messages.base import (
from langchain_core.messages.tool import (
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils._merge import merge_dicts, merge_lists
from langchain_core.utils.json import (
class AIMessageChunk(AIMessage, BaseMessageChunk):
    """Message chunk from an AI."""
    type: Literal['AIMessageChunk'] = 'AIMessageChunk'
    tool_call_chunks: List[ToolCallChunk] = []
    'If provided, tool call chunks associated with the message.'

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'messages']

    @property
    def lc_attributes(self) -> Dict:
        """Attrs to be serialized even if they are derived from other init args."""
        return {'tool_calls': self.tool_calls, 'invalid_tool_calls': self.invalid_tool_calls}

    @root_validator()
    def init_tool_calls(cls, values: dict) -> dict:
        if not values['tool_call_chunks']:
            values['tool_calls'] = []
            values['invalid_tool_calls'] = []
            return values
        tool_calls = []
        invalid_tool_calls = []
        for chunk in values['tool_call_chunks']:
            try:
                args_ = parse_partial_json(chunk['args'])
                if isinstance(args_, dict):
                    tool_calls.append(ToolCall(name=chunk['name'] or '', args=args_, id=chunk['id']))
                else:
                    raise ValueError('Malformed args.')
            except Exception:
                invalid_tool_calls.append(InvalidToolCall(name=chunk['name'], args=chunk['args'], id=chunk['id'], error='Malformed args.'))
        values['tool_calls'] = tool_calls
        values['invalid_tool_calls'] = invalid_tool_calls
        return values

    def __add__(self, other: Any) -> BaseMessageChunk:
        if isinstance(other, AIMessageChunk):
            if self.example != other.example:
                raise ValueError('Cannot concatenate AIMessageChunks with different example values.')
            content = merge_content(self.content, other.content)
            additional_kwargs = merge_dicts(self.additional_kwargs, other.additional_kwargs)
            response_metadata = merge_dicts(self.response_metadata, other.response_metadata)
            if self.tool_call_chunks or other.tool_call_chunks:
                raw_tool_calls = merge_lists(self.tool_call_chunks, other.tool_call_chunks)
                if raw_tool_calls:
                    tool_call_chunks = [ToolCallChunk(name=rtc.get('name'), args=rtc.get('args'), index=rtc.get('index'), id=rtc.get('id')) for rtc in raw_tool_calls]
                else:
                    tool_call_chunks = []
            else:
                tool_call_chunks = []
            return self.__class__(example=self.example, content=content, additional_kwargs=additional_kwargs, tool_call_chunks=tool_call_chunks, response_metadata=response_metadata, id=self.id)
        return super().__add__(other)