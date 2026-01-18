from dataclasses import dataclass
from typing import List, Literal, Optional
from .base import BaseInferenceType
@dataclass
class TextGenerationOutputDetails(BaseInferenceType):
    """When enabled, details about the generation"""
    finish_reason: 'TextGenerationFinishReason'
    'The reason why the generation was stopped.'
    generated_tokens: int
    'The number of generated tokens'
    prefill: List[TextGenerationPrefillToken]
    tokens: List[TextGenerationOutputToken]
    'The generated tokens and associated details'
    best_of_sequences: Optional[List[TextGenerationOutputSequenceDetails]] = None
    'Details about additional sequences when best_of is provided'
    seed: Optional[int] = None
    'The random seed used for generation'
    top_tokens: Optional[List[List[TextGenerationOutputToken]]] = None
    'Most likely tokens'