from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from .base import BaseInferenceType
@dataclass
class ZeroShotObjectDetectionBoundingBox(BaseInferenceType):
    """The predicted bounding box. Coordinates are relative to the top left corner of the input
    image.
    """
    xmax: int
    xmin: int
    ymax: int
    ymin: int