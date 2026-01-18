from typing_extensions import Literal
from ...._models import BaseModel
class FileCitationAnnotation(BaseModel):
    end_index: int
    file_citation: FileCitation
    start_index: int
    text: str
    'The text in the message content that needs to be replaced.'
    type: Literal['file_citation']
    'Always `file_citation`.'