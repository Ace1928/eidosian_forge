from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from .base import BaseInferenceType
@dataclass
class TableQuestionAnsweringOutputElement(BaseInferenceType):
    """Outputs of inference for the Table Question Answering task"""
    answer: str
    'The answer of the question given the table. If there is an aggregator, the answer will be\n    preceded by `AGGREGATOR >`.\n    '
    cells: List[str]
    'List of strings made up of the answer cell values.'
    coordinates: List[List[int]]
    'Coordinates of the cells of the answers.'
    aggregator: Optional[str] = None
    'If the model has an aggregator, this returns the aggregator.'