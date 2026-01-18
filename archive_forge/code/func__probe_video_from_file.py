import math
import warnings
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Union
import torch
from ..extension import _load_library
def _probe_video_from_file(filename: str) -> VideoMetaData:
    """
    Probe a video file and return VideoMetaData with info about the video
    """
    result = torch.ops.video_reader.probe_video_from_file(filename)
    vtimebase, vfps, vduration, atimebase, asample_rate, aduration = result
    info = _fill_info(vtimebase, vfps, vduration, atimebase, asample_rate, aduration)
    return info