from __future__ import annotations
from typing import Union, Iterable, Optional
from typing_extensions import Literal, Required, TypedDict
class RunSubmitToolOutputsParamsBase(TypedDict, total=False):
    thread_id: Required[str]
    tool_outputs: Required[Iterable[ToolOutput]]
    'A list of tools for which the outputs are being submitted.'