from __future__ import annotations
from typing import Optional
from typing_extensions import Required, TypedDict
class RunUpdateParams(TypedDict, total=False):
    thread_id: Required[str]
    metadata: Optional[object]
    'Set of 16 key-value pairs that can be attached to an object.\n\n    This can be useful for storing additional information about the object in a\n    structured format. Keys can be a maximum of 64 characters long and values can be\n    a maxium of 512 characters long.\n    '