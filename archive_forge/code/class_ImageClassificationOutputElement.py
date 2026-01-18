from dataclasses import dataclass
from typing import Any, Literal, Optional
from .base import BaseInferenceType
@dataclass
class ImageClassificationOutputElement(BaseInferenceType):
    """Outputs of inference for the Image Classification task"""
    label: str
    'The predicted class label.'
    score: float
    'The corresponding probability.'