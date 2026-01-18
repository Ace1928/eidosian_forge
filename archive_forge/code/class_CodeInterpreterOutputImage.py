from typing import List, Union
from typing_extensions import Literal, Annotated
from ....._utils import PropertyInfo
from ....._models import BaseModel
class CodeInterpreterOutputImage(BaseModel):
    image: CodeInterpreterOutputImageImage
    type: Literal['image']
    'Always `image`.'