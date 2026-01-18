from dataclasses import dataclass
from typing import Any, List, Optional, Union
from .base import BaseInferenceType
@dataclass
class DocumentQuestionAnsweringOutputElement(BaseInferenceType):
    """Outputs of inference for the Document Question Answering task"""
    answer: str
    'The answer to the question.'
    end: int
    'The end word index of the answer (in the OCR’d version of the input or provided word\n    boxes).\n    '
    score: float
    'The probability associated to the answer.'
    start: int
    'The start word index of the answer (in the OCR’d version of the input or provided word\n    boxes).\n    '
    words: List[int]
    'The index of each word/box pair that is in the answer'