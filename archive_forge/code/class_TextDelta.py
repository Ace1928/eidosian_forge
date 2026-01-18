from typing import List, Optional
from ...._models import BaseModel
from .annotation_delta import AnnotationDelta
class TextDelta(BaseModel):
    annotations: Optional[List[AnnotationDelta]] = None
    value: Optional[str] = None
    'The data that makes up the text.'