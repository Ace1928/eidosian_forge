from typing_extensions import Literal
from ...._models import BaseModel
class FileDeleteResponse(BaseModel):
    id: str
    deleted: bool
    object: Literal['assistant.file.deleted']