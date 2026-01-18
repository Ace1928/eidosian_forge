from __future__ import annotations
from typing_extensions import Literal, Required, TypedDict
class ChatCompletionContentPartImageParam(TypedDict, total=False):
    image_url: Required[ImageURL]
    type: Required[Literal['image_url']]
    'The type of the content part.'