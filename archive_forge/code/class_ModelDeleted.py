from .._models import BaseModel
class ModelDeleted(BaseModel):
    id: str
    deleted: bool
    object: str