from __future__ import annotations
from typing_extensions import Required, TypedDict
class AssistantToolChoiceFunctionParam(TypedDict, total=False):
    name: Required[str]
    'The name of the function to call.'