from dataclasses import dataclass
from typing import Any, List, Optional
from .base import BaseInferenceType
@dataclass
class ImageToImageOutput(BaseInferenceType):
    """Outputs of inference for the Image To Image task"""
    image: Any
    'The output image'