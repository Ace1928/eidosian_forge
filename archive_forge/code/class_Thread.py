from typing import Optional
from typing_extensions import Literal
from ..._models import BaseModel
class Thread(BaseModel):
    id: str
    'The identifier, which can be referenced in API endpoints.'
    created_at: int
    'The Unix timestamp (in seconds) for when the thread was created.'
    metadata: Optional[object] = None
    'Set of 16 key-value pairs that can be attached to an object.\n\n    This can be useful for storing additional information about the object in a\n    structured format. Keys can be a maximum of 64 characters long and values can be\n    a maxium of 512 characters long.\n    '
    object: Literal['thread']
    'The object type, which is always `thread`.'