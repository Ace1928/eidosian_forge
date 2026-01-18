from typing_extensions import Literal
from ..._models import BaseModel
class RetrievalTool(BaseModel):
    type: Literal['retrieval']
    'The type of tool being defined: `retrieval`'