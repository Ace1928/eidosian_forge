from dataclasses import dataclass
from typing import Any, List, Optional
from .base import BaseInferenceType
@dataclass
class ZeroShotImageClassificationInputData(BaseInferenceType):
    """The input image data, with candidate labels"""
    candidate_labels: List[str]
    'The candidate labels for this image'
    image: Any
    'The image data to classify'