from dataclasses import dataclass
from typing import Any, Optional
from .base import BaseInferenceType
@dataclass
class VisualQuestionAnsweringInput(BaseInferenceType):
    """Inputs for Visual Question Answering inference"""
    inputs: VisualQuestionAnsweringInputData
    'One (image, question) pair to answer'
    parameters: Optional[VisualQuestionAnsweringParameters] = None
    'Additional inference parameters'