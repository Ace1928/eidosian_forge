from __future__ import annotations
from typing import List, Optional
from typing_extensions import Literal, Required, TypedDict
class VectorStoreCreateParams(TypedDict, total=False):
    expires_after: ExpiresAfter
    'The expiration policy for a vector store.'
    file_ids: List[str]
    '\n    A list of [File](https://platform.openai.com/docs/api-reference/files) IDs that\n    the vector store should use. Useful for tools like `file_search` that can access\n    files.\n    '
    metadata: Optional[object]
    'Set of 16 key-value pairs that can be attached to an object.\n\n    This can be useful for storing additional information about the object in a\n    structured format. Keys can be a maximum of 64 characters long and values can be\n    a maxium of 512 characters long.\n    '
    name: str
    'The name of the vector store.'