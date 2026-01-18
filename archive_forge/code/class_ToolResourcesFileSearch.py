from typing import List, Optional
from typing_extensions import Literal
from ..._models import BaseModel
class ToolResourcesFileSearch(BaseModel):
    vector_store_ids: Optional[List[str]] = None
    '\n    The\n    [vector store](https://platform.openai.com/docs/api-reference/vector-stores/object)\n    attached to this thread. There can be a maximum of 1 vector store attached to\n    the thread.\n    '