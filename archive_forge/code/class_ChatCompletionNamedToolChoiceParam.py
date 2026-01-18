from __future__ import annotations
from typing_extensions import Literal, Required, TypedDict
class ChatCompletionNamedToolChoiceParam(TypedDict, total=False):
    function: Required[Function]
    type: Required[Literal['function']]
    'The type of the tool. Currently, only `function` is supported.'