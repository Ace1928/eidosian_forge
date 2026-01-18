from __future__ import annotations
from typing import List, Union
from typing_extensions import Literal, Required, TypedDict
class ModerationCreateParams(TypedDict, total=False):
    input: Required[Union[str, List[str]]]
    'The input text to classify'
    model: Union[str, Literal['text-moderation-latest', 'text-moderation-stable']]
    '\n    Two content moderations models are available: `text-moderation-stable` and\n    `text-moderation-latest`.\n\n    The default is `text-moderation-latest` which will be automatically upgraded\n    over time. This ensures you are always using our most accurate model. If you use\n    `text-moderation-stable`, we will provide advanced notice before updating the\n    model. Accuracy of `text-moderation-stable` may be slightly lower than for\n    `text-moderation-latest`.\n    '