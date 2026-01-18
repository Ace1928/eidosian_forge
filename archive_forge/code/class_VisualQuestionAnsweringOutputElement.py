from dataclasses import dataclass
from typing import Any, Optional
from .base import BaseInferenceType
@dataclass
class VisualQuestionAnsweringOutputElement(BaseInferenceType):
    """Outputs of inference for the Visual Question Answering task"""
    label: Any
    score: float
    'The associated score / probability'
    answer: Optional[str] = None
    'The answer to the question'