from dataclasses import dataclass
from typing import Any, Literal, Optional
from .base import BaseInferenceType
@dataclass
class AudioClassificationOutputElement(BaseInferenceType):
    """Outputs for Audio Classification inference"""
    label: str
    'The predicted class label.'
    score: float
    'The corresponding probability.'