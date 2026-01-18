from dataclasses import dataclass
from typing import List, Literal, Optional
from .base import BaseInferenceType
@dataclass
class TextGenerationStreamOutput(BaseInferenceType):
    """Text Generation Stream Output"""
    token: TextGenerationOutputToken
    'Generated token.'
    details: Optional[TextGenerationStreamDetails] = None
    'Generation details. Only available when the generation is finished.'
    generated_text: Optional[str] = None
    'The complete generated text. Only available when the generation is finished.'
    index: Optional[int] = None
    'The token index within the stream. Optional to support older clients that omit it.'