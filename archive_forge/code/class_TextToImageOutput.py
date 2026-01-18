from dataclasses import dataclass
from typing import Any, List, Optional
from .base import BaseInferenceType
@dataclass
class TextToImageOutput(BaseInferenceType):
    """Outputs of inference for the Text To Image task"""
    image: Any
    'The generated image'