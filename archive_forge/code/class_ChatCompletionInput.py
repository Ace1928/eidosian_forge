from dataclasses import dataclass
from typing import List, Literal, Optional, Union
from .base import BaseInferenceType
@dataclass
class ChatCompletionInput(BaseInferenceType):
    """Inputs for ChatCompletion inference"""
    messages: List[ChatCompletionInputMessage]
    frequency_penalty: Optional[float] = None
    "Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing\n    frequency in the text so far, decreasing the model's likelihood to repeat the same line\n    verbatim.\n    "
    max_tokens: Optional[int] = None
    'The maximum number of tokens that can be generated in the chat completion.'
    seed: Optional[int] = None
    'The random sampling seed.'
    stop: Optional[Union[List[str], str]] = None
    'Stop generating tokens if a stop token is generated.'
    stream: Optional[bool] = None
    'If set, partial message deltas will be sent.'
    temperature: Optional[float] = None
    'The value used to modulate the logits distribution.'
    top_p: Optional[float] = None
    'If set to < 1, only the smallest set of most probable tokens with probabilities that add\n    up to `top_p` or higher are kept for generation.\n    '