from dataclasses import dataclass
from typing import Any, List, Literal, Optional
from .base import BaseInferenceType
@dataclass
class TokenClassificationOutputElement(BaseInferenceType):
    """Outputs of inference for the Token Classification task"""
    label: Any
    score: float
    'The associated score / probability'
    end: Optional[int] = None
    'The character position in the input where this group ends.'
    entity_group: Optional[str] = None
    'The predicted label for that group of tokens'
    start: Optional[int] = None
    'The character position in the input where this group begins.'
    word: Optional[str] = None
    'The corresponding text'