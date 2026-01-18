from typing_extensions import Literal
from ...._models import BaseModel
class FileCitation(BaseModel):
    file_id: str
    'The ID of the specific File the citation is from.'
    quote: str
    'The specific quote in the file.'