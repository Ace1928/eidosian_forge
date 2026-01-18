from dataclasses import dataclass
from typing import List, Literal, Optional
from .base import BaseInferenceType
@dataclass
class TextGenerationPrefillToken(BaseInferenceType):
    id: int
    logprob: float
    text: str
    'The text associated with that token'