from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional
from .base import BaseInferenceType
@dataclass
class Text2TextGenerationOutput(BaseInferenceType):
    """Outputs of inference for the Text2text Generation task"""
    generated_text: Any
    text2_text_generation_output_generated_text: Optional[str] = None
    'The generated text.'