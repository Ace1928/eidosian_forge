from dataclasses import dataclass
from typing import Any, Dict, Optional
from .base import BaseInferenceType
@dataclass
class DepthEstimationOutput(BaseInferenceType):
    """Outputs of inference for the Depth Estimation task"""
    depth: Any
    'The predicted depth as an image'
    predicted_depth: Any
    'The predicted depth as a tensor'