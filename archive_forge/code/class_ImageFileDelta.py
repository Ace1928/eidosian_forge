from typing import Optional
from ...._models import BaseModel
class ImageFileDelta(BaseModel):
    file_id: Optional[str] = None
    '\n    The [File](https://platform.openai.com/docs/api-reference/files) ID of the image\n    in the message content.\n    '