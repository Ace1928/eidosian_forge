from dataclasses import dataclass
from typing import Any, List, Optional
from .base import BaseInferenceType
@dataclass
class ImageToImageInput(BaseInferenceType):
    """Inputs for Image To Image inference"""
    inputs: Any
    'The input image data'
    parameters: Optional[ImageToImageParameters] = None
    'Additional inference parameters'