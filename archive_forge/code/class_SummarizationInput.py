from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional
from .base import BaseInferenceType
@dataclass
class SummarizationInput(BaseInferenceType):
    """Inputs for Summarization inference
    Inputs for Text2text Generation inference
    """
    inputs: str
    'The input text data'
    parameters: Optional[SummarizationGenerationParameters] = None
    'Additional inference parameters'