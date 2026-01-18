from dataclasses import dataclass
from typing import Any, Literal, Optional
from .base import BaseInferenceType
@dataclass
class ImageSegmentationInput(BaseInferenceType):
    """Inputs for Image Segmentation inference"""
    inputs: Any
    'The input image data'
    parameters: Optional[ImageSegmentationParameters] = None
    'Additional inference parameters'