from typing import Any, Dict, List, Literal
from langchain_core.messages.base import (
from langchain_core.messages.tool import (
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils._merge import merge_dicts, merge_lists
from langchain_core.utils.json import (
class AIMessage(BaseMessage):
    """Message from an AI."""
    example: bool = False
    'Whether this Message is being passed in to the model as part of an example \n        conversation.\n    '
    tool_calls: List[ToolCall] = []
    'If provided, tool calls associated with the message.'
    invalid_tool_calls: List[InvalidToolCall] = []
    'If provided, tool calls with parsing errors associated with the message.'
    type: Literal['ai'] = 'ai'

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'messages']

    @property
    def lc_attributes(self) -> Dict:
        """Attrs to be serialized even if they are derived from other init args."""
        return {'tool_calls': self.tool_calls, 'invalid_tool_calls': self.invalid_tool_calls}

    @root_validator()
    def _backwards_compat_tool_calls(cls, values: dict) -> dict:
        raw_tool_calls = values.get('additional_kwargs', {}).get('tool_calls')
        tool_calls = values.get('tool_calls') or values.get('invalid_tool_calls') or values.get('tool_call_chunks')
        if raw_tool_calls and (not tool_calls):
            try:
                if issubclass(cls, AIMessageChunk):
                    values['tool_call_chunks'] = default_tool_chunk_parser(raw_tool_calls)
                else:
                    tool_calls, invalid_tool_calls = default_tool_parser(raw_tool_calls)
                    values['tool_calls'] = tool_calls
                    values['invalid_tool_calls'] = invalid_tool_calls
            except Exception:
                pass
        return values