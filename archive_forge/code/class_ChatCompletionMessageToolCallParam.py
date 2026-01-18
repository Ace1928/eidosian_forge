from __future__ import annotations
from typing_extensions import Literal, Required, TypedDict
class ChatCompletionMessageToolCallParam(TypedDict, total=False):
    id: Required[str]
    'The ID of the tool call.'
    function: Required[Function]
    'The function that the model called.'
    type: Required[Literal['function']]
    'The type of the tool. Currently, only `function` is supported.'