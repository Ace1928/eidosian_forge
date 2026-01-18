from __future__ import annotations
from typing import Union, Optional
from typing_extensions import Literal, Required, TypedDict
from .._types import FileTypes
class ImageEditParams(TypedDict, total=False):
    image: Required[FileTypes]
    'The image to edit.\n\n    Must be a valid PNG file, less than 4MB, and square. If mask is not provided,\n    image must have transparency, which will be used as the mask.\n    '
    prompt: Required[str]
    'A text description of the desired image(s).\n\n    The maximum length is 1000 characters.\n    '
    mask: FileTypes
    'An additional image whose fully transparent areas (e.g.\n\n    where alpha is zero) indicate where `image` should be edited. Must be a valid\n    PNG file, less than 4MB, and have the same dimensions as `image`.\n    '
    model: Union[str, Literal['dall-e-2'], None]
    'The model to use for image generation.\n\n    Only `dall-e-2` is supported at this time.\n    '
    n: Optional[int]
    'The number of images to generate. Must be between 1 and 10.'
    response_format: Optional[Literal['url', 'b64_json']]
    'The format in which the generated images are returned.\n\n    Must be one of `url` or `b64_json`. URLs are only valid for 60 minutes after the\n    image has been generated.\n    '
    size: Optional[Literal['256x256', '512x512', '1024x1024']]
    'The size of the generated images.\n\n    Must be one of `256x256`, `512x512`, or `1024x1024`.\n    '
    user: str
    '\n    A unique identifier representing your end-user, which can help OpenAI to monitor\n    and detect abuse.\n    [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).\n    '