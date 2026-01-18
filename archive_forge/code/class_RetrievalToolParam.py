from __future__ import annotations
from typing_extensions import Literal, Required, TypedDict
class RetrievalToolParam(TypedDict, total=False):
    type: Required[Literal['retrieval']]
    'The type of tool being defined: `retrieval`'