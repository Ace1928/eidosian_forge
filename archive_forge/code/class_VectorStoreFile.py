from typing import Optional
from typing_extensions import Literal
from ...._models import BaseModel
class VectorStoreFile(BaseModel):
    id: str
    'The identifier, which can be referenced in API endpoints.'
    created_at: int
    'The Unix timestamp (in seconds) for when the vector store file was created.'
    last_error: Optional[LastError] = None
    'The last error associated with this vector store file.\n\n    Will be `null` if there are no errors.\n    '
    object: Literal['vector_store.file']
    'The object type, which is always `vector_store.file`.'
    status: Literal['in_progress', 'completed', 'cancelled', 'failed']
    '\n    The status of the vector store file, which can be either `in_progress`,\n    `completed`, `cancelled`, or `failed`. The status `completed` indicates that the\n    vector store file is ready for use.\n    '
    vector_store_id: str
    '\n    The ID of the\n    [vector store](https://platform.openai.com/docs/api-reference/vector-stores/object)\n    that the [File](https://platform.openai.com/docs/api-reference/files) is\n    attached to.\n    '