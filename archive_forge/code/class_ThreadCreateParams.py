from __future__ import annotations
from typing import List, Iterable, Optional
from typing_extensions import Literal, Required, TypedDict
class ThreadCreateParams(TypedDict, total=False):
    messages: Iterable[Message]
    '\n    A list of [messages](https://platform.openai.com/docs/api-reference/messages) to\n    start the thread with.\n    '
    metadata: Optional[object]
    'Set of 16 key-value pairs that can be attached to an object.\n\n    This can be useful for storing additional information about the object in a\n    structured format. Keys can be a maximum of 64 characters long and values can be\n    a maxium of 512 characters long.\n    '