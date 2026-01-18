from dataclasses import dataclass
from typing import Any, Literal, Optional
from .base import BaseInferenceType
@dataclass
class VideoClassificationOutputElement(BaseInferenceType):
    """Outputs of inference for the Video Classification task"""
    label: str
    'The predicted class label.'
    score: float
    'The corresponding probability.'