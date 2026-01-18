from __future__ import annotations
from typing import List, Union, Iterable, Optional
from typing_extensions import Literal, Required, TypedDict
from .function_tool_param import FunctionToolParam
from .retrieval_tool_param import RetrievalToolParam
from .code_interpreter_tool_param import CodeInterpreterToolParam
class ThreadCreateAndRunParamsNonStreaming(ThreadCreateAndRunParamsBase):
    stream: Optional[Literal[False]]
    '\n    If `true`, returns a stream of events that happen during the Run as server-sent\n    events, terminating when the Run enters a terminal state with a `data: [DONE]`\n    message.\n    '