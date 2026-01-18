from __future__ import annotations
from typing_extensions import Literal, Required, TypedDict
class ImageURL(TypedDict, total=False):
    url: Required[str]
    'Either a URL of the image or the base64 encoded image data.'
    detail: Literal['auto', 'low', 'high']
    'Specifies the detail level of the image.\n\n    Learn more in the\n    [Vision guide](https://platform.openai.com/docs/guides/vision/low-or-high-fidelity-image-understanding).\n    '