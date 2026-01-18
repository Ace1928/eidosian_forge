from dataclasses import dataclass
from typing import Optional
from .base import BaseInferenceType
@dataclass
class QuestionAnsweringInputData(BaseInferenceType):
    """One (context, question) pair to answer"""
    context: str
    'The context to be used for answering the question'
    question: str
    'The question to be answered'