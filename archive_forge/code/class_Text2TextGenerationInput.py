from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional
from .base import BaseInferenceType
@dataclass
class Text2TextGenerationInput(BaseInferenceType):
    """Inputs for Text2text Generation inference"""
    inputs: str
    'The input text data'
    parameters: Optional[Text2TextGenerationParameters] = None
    'Additional inference parameters'