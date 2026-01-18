from dataclasses import dataclass
from typing import Any, Literal, Optional, Union
from .base import BaseInferenceType
@dataclass
class ImageToTextParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Image To Text
    """
    generate: Optional[ImageToTextGenerationParameters] = None
    'Parametrization of the text generation process'
    max_new_tokens: Optional[int] = None
    'The amount of maximum tokens to generate.'