from dataclasses import dataclass
from typing import Any, Literal, Optional
from .base import BaseInferenceType
@dataclass
class AudioClassificationParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Audio Classification
    """
    function_to_apply: Optional['ClassificationOutputTransform'] = None
    top_k: Optional[int] = None
    'When specified, limits the output to the top K most probable classes.'