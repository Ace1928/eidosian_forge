from dataclasses import dataclass
from typing import List, Literal, Optional
from .base import BaseInferenceType
@dataclass
class TextGenerationInput(BaseInferenceType):
    """Inputs for Text Generation inference"""
    inputs: str
    'The text to initialize generation with'
    parameters: Optional[TextGenerationParameters] = None
    'Additional inference parameters'
    stream: Optional[bool] = None
    'Whether to stream output tokens'