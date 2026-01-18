from __future__ import annotations
from typing_extensions import Literal, Required, TypedDict
class CodeInterpreterToolParam(TypedDict, total=False):
    type: Required[Literal['code_interpreter']]
    'The type of tool being defined: `code_interpreter`'