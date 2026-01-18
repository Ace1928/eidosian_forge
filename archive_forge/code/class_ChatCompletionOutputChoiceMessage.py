from dataclasses import dataclass
from typing import List, Literal, Optional, Union
from .base import BaseInferenceType
@dataclass
class ChatCompletionOutputChoiceMessage(BaseInferenceType):
    content: str
    'The content of the chat completion message.'
    role: 'ChatCompletionMessageRole'