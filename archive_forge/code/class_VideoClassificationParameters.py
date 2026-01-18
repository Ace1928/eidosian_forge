from dataclasses import dataclass
from typing import Any, Literal, Optional
from .base import BaseInferenceType
@dataclass
class VideoClassificationParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Video Classification
    """
    frame_sampling_rate: Optional[int] = None
    'The sampling rate used to select frames from the video.'
    function_to_apply: Optional['ClassificationOutputTransform'] = None
    num_frames: Optional[int] = None
    'The number of sampled frames to consider for classification.'
    top_k: Optional[int] = None
    'When specified, limits the output to the top K most probable classes.'