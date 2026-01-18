from typing_extensions import Literal
from ...._models import BaseModel
class FilePathAnnotation(BaseModel):
    end_index: int
    file_path: FilePath
    start_index: int
    text: str
    'The text in the message content that needs to be replaced.'
    type: Literal['file_path']
    'Always `file_path`.'