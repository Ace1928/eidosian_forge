from typing_extensions import Literal
from .._models import BaseModel
class FileDeleted(BaseModel):
    id: str
    deleted: bool
    object: Literal['file']