from __future__ import annotations
from typing import Dict, List, Union, Iterable, Optional
from typing_extensions import Literal, Required, TypedDict
from ...types import shared_params
from .chat_completion_tool_param import ChatCompletionToolParam
from .chat_completion_message_param import ChatCompletionMessageParam
from .chat_completion_tool_choice_option_param import ChatCompletionToolChoiceOptionParam
from .chat_completion_function_call_option_param import ChatCompletionFunctionCallOptionParam
class ResponseFormat(TypedDict, total=False):
    type: Literal['text', 'json_object']
    'Must be one of `text` or `json_object`.'