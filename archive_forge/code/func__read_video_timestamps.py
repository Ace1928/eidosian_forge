import math
import warnings
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Union
import torch
from ..extension import _load_library
def _read_video_timestamps(filename: str, pts_unit: str='pts') -> Tuple[Union[List[int], List[Fraction]], Optional[float]]:
    if pts_unit == 'pts':
        warnings.warn("The pts_unit 'pts' gives wrong results and will be removed in a " + "follow-up version. Please use pts_unit 'sec'.")
    pts: Union[List[int], List[Fraction]]
    pts, _, info = _read_video_timestamps_from_file(filename)
    if pts_unit == 'sec':
        video_time_base = Fraction(info.video_timebase.numerator, info.video_timebase.denominator)
        pts = [x * video_time_base for x in pts]
    video_fps = info.video_fps if info.has_video else None
    return (pts, video_fps)