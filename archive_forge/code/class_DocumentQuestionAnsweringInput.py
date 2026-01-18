from dataclasses import dataclass
from typing import Any, List, Optional, Union
from .base import BaseInferenceType
@dataclass
class DocumentQuestionAnsweringInput(BaseInferenceType):
    """Inputs for Document Question Answering inference"""
    inputs: DocumentQuestionAnsweringInputData
    'One (document, question) pair to answer'
    parameters: Optional[DocumentQuestionAnsweringParameters] = None
    'Additional inference parameters'