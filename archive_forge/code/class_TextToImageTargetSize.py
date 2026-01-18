from dataclasses import dataclass
from typing import Any, List, Optional
from .base import BaseInferenceType
@dataclass
class TextToImageTargetSize(BaseInferenceType):
    """The size in pixel of the output image"""
    height: int
    width: int