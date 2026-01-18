from dataclasses import dataclass
from typing import Any, Literal, Optional
from .base import BaseInferenceType
@dataclass
class VideoClassificationInput(BaseInferenceType):
    """Inputs for Video Classification inference"""
    inputs: Any
    'The input video data'
    parameters: Optional[VideoClassificationParameters] = None
    'Additional inference parameters'