from dataclasses import dataclass
from typing import List, Literal, Optional, Union
from .base import BaseInferenceType
@dataclass
class ChatCompletionOutputChoice(BaseInferenceType):
    finish_reason: 'ChatCompletionFinishReason'
    'The reason why the generation was stopped.'
    index: int
    'The index of the choice in the list of choices.'
    message: ChatCompletionOutputChoiceMessage