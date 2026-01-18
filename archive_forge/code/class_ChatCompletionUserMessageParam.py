from __future__ import annotations
from typing import Union, Iterable
from typing_extensions import Literal, Required, TypedDict
from .chat_completion_content_part_param import ChatCompletionContentPartParam
class ChatCompletionUserMessageParam(TypedDict, total=False):
    content: Required[Union[str, Iterable[ChatCompletionContentPartParam]]]
    'The contents of the user message.'
    role: Required[Literal['user']]
    'The role of the messages author, in this case `user`.'
    name: str
    'An optional name for the participant.\n\n    Provides the model information to differentiate between participants of the same\n    role.\n    '