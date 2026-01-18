from typing import Any, Dict, List, Literal
from langchain_core.messages.base import (
from langchain_core.messages.tool import (
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils._merge import merge_dicts, merge_lists
from langchain_core.utils.json import (
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