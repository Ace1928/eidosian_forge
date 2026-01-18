from typing_extensions import Literal
from ..._models import BaseModel
class VectorStoreDeleted(BaseModel):
    id: str
    deleted: bool
    object: Literal['vector_store.deleted']