from dataclasses import dataclass
from typing import List, Literal, Optional
from .base import BaseInferenceType
@dataclass
class TextGenerationOutput(BaseInferenceType):
    """Outputs for Text Generation inference"""
    generated_text: str
    'The generated text'
    details: Optional[TextGenerationOutputDetails] = None
    'When enabled, details about the generation'