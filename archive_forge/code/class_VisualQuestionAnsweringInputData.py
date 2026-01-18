from dataclasses import dataclass
from typing import Any, Optional
from .base import BaseInferenceType
@dataclass
class VisualQuestionAnsweringInputData(BaseInferenceType):
    """One (image, question) pair to answer"""
    image: Any
    'The image.'
    question: Any
    'The question to answer based on the image.'