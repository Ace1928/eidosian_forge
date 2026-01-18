from __future__ import annotations
from typing import List, Iterable, Optional
from typing_extensions import Required, TypedDict
from .assistant_tool_param import AssistantToolParam
A list of tool enabled on the assistant.

    There can be a maximum of 128 tools per assistant. Tools can be of types
    `code_interpreter`, `retrieval`, or `function`.
    