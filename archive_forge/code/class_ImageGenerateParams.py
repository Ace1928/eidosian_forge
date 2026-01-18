from __future__ import annotations
from typing import Union, Optional
from typing_extensions import Literal, Required, TypedDict
class ImageGenerateParams(TypedDict, total=False):
    prompt: Required[str]
    'A text description of the desired image(s).\n\n    The maximum length is 1000 characters for `dall-e-2` and 4000 characters for\n    `dall-e-3`.\n    '
    model: Union[str, Literal['dall-e-2', 'dall-e-3'], None]
    'The model to use for image generation.'
    n: Optional[int]
    'The number of images to generate.\n\n    Must be between 1 and 10. For `dall-e-3`, only `n=1` is supported.\n    '
    quality: Literal['standard', 'hd']
    'The quality of the image that will be generated.\n\n    `hd` creates images with finer details and greater consistency across the image.\n    This param is only supported for `dall-e-3`.\n    '
    response_format: Optional[Literal['url', 'b64_json']]
    'The format in which the generated images are returned.\n\n    Must be one of `url` or `b64_json`. URLs are only valid for 60 minutes after the\n    image has been generated.\n    '
    size: Optional[Literal['256x256', '512x512', '1024x1024', '1792x1024', '1024x1792']]
    'The size of the generated images.\n\n    Must be one of `256x256`, `512x512`, or `1024x1024` for `dall-e-2`. Must be one\n    of `1024x1024`, `1792x1024`, or `1024x1792` for `dall-e-3` models.\n    '
    style: Optional[Literal['vivid', 'natural']]
    'The style of the generated images.\n\n    Must be one of `vivid` or `natural`. Vivid causes the model to lean towards\n    generating hyper-real and dramatic images. Natural causes the model to produce\n    more natural, less hyper-real looking images. This param is only supported for\n    `dall-e-3`.\n    '
    user: str
    '\n    A unique identifier representing your end-user, which can help OpenAI to monitor\n    and detect abuse.\n    [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).\n    '