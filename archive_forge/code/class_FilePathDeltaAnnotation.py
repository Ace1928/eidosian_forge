from typing import Optional
from typing_extensions import Literal
from ...._models import BaseModel
class FilePathDeltaAnnotation(BaseModel):
    index: int
    'The index of the annotation in the text content part.'
    type: Literal['file_path']
    'Always `file_path`.'
    end_index: Optional[int] = None
    file_path: Optional[FilePath] = None
    start_index: Optional[int] = None
    text: Optional[str] = None
    'The text in the message content that needs to be replaced.'