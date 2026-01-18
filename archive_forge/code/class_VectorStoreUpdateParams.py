from __future__ import annotations
from typing import Optional
from typing_extensions import Literal, Required, TypedDict
class VectorStoreUpdateParams(TypedDict, total=False):
    expires_after: Optional[ExpiresAfter]
    'The expiration policy for a vector store.'
    metadata: Optional[object]
    'Set of 16 key-value pairs that can be attached to an object.\n\n    This can be useful for storing additional information about the object in a\n    structured format. Keys can be a maximum of 64 characters long and values can be\n    a maxium of 512 characters long.\n    '
    name: Optional[str]
    'The name of the vector store.'