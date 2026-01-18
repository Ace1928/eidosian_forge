from dataclasses import dataclass
from typing import List, Literal, Optional
from .base import BaseInferenceType
@dataclass
class TextGenerationOutputSequenceDetails(BaseInferenceType):
    finish_reason: 'TextGenerationFinishReason'
    generated_text: str
    'The generated text'
    generated_tokens: int
    'The number of generated tokens'
    prefill: List[TextGenerationPrefillToken]
    tokens: List[TextGenerationOutputToken]
    'The generated tokens and associated details'
    seed: Optional[int] = None
    'The random seed used for generation'
    top_tokens: Optional[List[List[TextGenerationOutputToken]]] = None
    'Most likely tokens'