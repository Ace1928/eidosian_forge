from __future__ import annotations
from typing import Union, Iterable, Optional
from typing_extensions import Literal, Required, TypedDict
class ToolOutput(TypedDict, total=False):
    output: str
    'The output of the tool call to be submitted to continue the run.'
    tool_call_id: str
    '\n    The ID of the tool call in the `required_action` object within the run object\n    the output is being submitted for.\n    '