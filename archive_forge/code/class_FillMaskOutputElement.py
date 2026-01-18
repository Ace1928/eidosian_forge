from dataclasses import dataclass
from typing import Any, List, Optional
from .base import BaseInferenceType
@dataclass
class FillMaskOutputElement(BaseInferenceType):
    """Outputs of inference for the Fill Mask task"""
    score: float
    'The corresponding probability'
    sequence: str
    'The corresponding input with the mask token prediction.'
    token: int
    'The predicted token id (to replace the masked one).'
    token_str: Any
    fill_mask_output_token_str: Optional[str] = None
    'The predicted token (to replace the masked one).'