import json
from typing import Any, Dict, List, Literal, Optional, Tuple
from typing_extensions import TypedDict
from langchain_core.messages.base import (
from langchain_core.utils._merge import merge_dicts
class ToolCallChunk(TypedDict):
    """A chunk of a tool call (e.g., as part of a stream).

    When merging ToolCallChunks (e.g., via AIMessageChunk.__add__),
    all string attributes are concatenated. Chunks are only merged if their
    values of `index` are equal and not None.

    Example:

    .. code-block:: python

        left_chunks = [ToolCallChunk(name="foo", args='{"a":', index=0)]
        right_chunks = [ToolCallChunk(name=None, args='1}', index=0)]
        (
            AIMessageChunk(content="", tool_call_chunks=left_chunks)
            + AIMessageChunk(content="", tool_call_chunks=right_chunks)
        ).tool_call_chunks == [ToolCallChunk(name='foo', args='{"a":1}', index=0)]

    Attributes:
        name: (str) if provided, a substring of the name of the tool to be called
        args: (str) if provided, a JSON substring of the arguments to the tool call
        id: (str) if provided, a substring of an identifier for the tool call
        index: (int) if provided, the index of the tool call in a sequence
    """
    name: Optional[str]
    args: Optional[str]
    id: Optional[str]
    index: Optional[int]