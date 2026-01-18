from typing import Optional
from typing_extensions import Literal
from ...._models import BaseModel
from .image_file_delta import ImageFileDelta
class ImageFileDeltaBlock(BaseModel):
    index: int
    'The index of the content part in the message.'
    type: Literal['image_file']
    'Always `image_file`.'
    image_file: Optional[ImageFileDelta] = None