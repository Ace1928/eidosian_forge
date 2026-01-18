from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from .base import BaseInferenceType
@dataclass
class ZeroShotObjectDetectionInput(BaseInferenceType):
    """Inputs for Zero Shot Object Detection inference"""
    inputs: ZeroShotObjectDetectionInputData
    'The input image data, with candidate labels'
    parameters: Optional[Dict[str, Any]] = None
    'Additional inference parameters'