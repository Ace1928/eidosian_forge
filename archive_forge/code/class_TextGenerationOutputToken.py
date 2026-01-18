from dataclasses import dataclass
from typing import List, Literal, Optional
from .base import BaseInferenceType
@dataclass
class TextGenerationOutputToken(BaseInferenceType):
    """Generated token."""
    id: int
    special: bool
    'Whether or not that token is a special one'
    text: str
    'The text associated with that token'
    logprob: Optional[float] = None