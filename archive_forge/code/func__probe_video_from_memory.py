import math
import warnings
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Union
import torch
from ..extension import _load_library
def _probe_video_from_memory(video_data: torch.Tensor) -> VideoMetaData:
    """
    Probe a video in memory and return VideoMetaData with info about the video
    This function is torchscriptable
    """
    if not isinstance(video_data, torch.Tensor):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The given buffer is not writable')
            video_data = torch.frombuffer(video_data, dtype=torch.uint8)
    result = torch.ops.video_reader.probe_video_from_memory(video_data)
    vtimebase, vfps, vduration, atimebase, asample_rate, aduration = result
    info = _fill_info(vtimebase, vfps, vduration, atimebase, asample_rate, aduration)
    return info