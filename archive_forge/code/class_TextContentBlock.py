from typing_extensions import Literal
from .text import Text
from ...._models import BaseModel
class TextContentBlock(BaseModel):
    text: Text
    type: Literal['text']
    'Always `text`.'