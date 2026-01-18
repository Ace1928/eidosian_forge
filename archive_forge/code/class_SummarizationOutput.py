from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional
from .base import BaseInferenceType
@dataclass
class SummarizationOutput(BaseInferenceType):
    """Outputs of inference for the Summarization task"""
    summary_text: str
    'The summarized text.'