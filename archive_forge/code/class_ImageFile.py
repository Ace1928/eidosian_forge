from ...._models import BaseModel
class ImageFile(BaseModel):
    file_id: str
    '\n    The [File](https://platform.openai.com/docs/api-reference/files) ID of the image\n    in the message content.\n    '