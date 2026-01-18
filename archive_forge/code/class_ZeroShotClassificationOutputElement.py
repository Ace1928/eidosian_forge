from dataclasses import dataclass
from typing import List, Optional
from .base import BaseInferenceType
@dataclass
class ZeroShotClassificationOutputElement(BaseInferenceType):
    """Outputs of inference for the Zero Shot Classification task"""
    label: str
    'The predicted class label.'
    score: float
    'The corresponding probability.'