from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from .base import BaseInferenceType
@dataclass
class SentenceSimilarityInput(BaseInferenceType):
    """Inputs for Sentence similarity inference"""
    inputs: SentenceSimilarityInputData
    parameters: Optional[Dict[str, Any]] = None
    'Additional inference parameters'