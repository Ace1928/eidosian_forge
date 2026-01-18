import math
import warnings
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Union
import torch
from ..extension import _load_library
def _fill_info(vtimebase: torch.Tensor, vfps: torch.Tensor, vduration: torch.Tensor, atimebase: torch.Tensor, asample_rate: torch.Tensor, aduration: torch.Tensor) -> VideoMetaData:
    """
    Build update VideoMetaData struct with info about the video
    """
    meta = VideoMetaData()
    if vtimebase.numel() > 0:
        meta.video_timebase = Timebase(int(vtimebase[0].item()), int(vtimebase[1].item()))
        timebase = vtimebase[0].item() / float(vtimebase[1].item())
        if vduration.numel() > 0:
            meta.has_video = True
            meta.video_duration = float(vduration.item()) * timebase
    if vfps.numel() > 0:
        meta.video_fps = float(vfps.item())
    if atimebase.numel() > 0:
        meta.audio_timebase = Timebase(int(atimebase[0].item()), int(atimebase[1].item()))
        timebase = atimebase[0].item() / float(atimebase[1].item())
        if aduration.numel() > 0:
            meta.has_audio = True
            meta.audio_duration = float(aduration.item()) * timebase
    if asample_rate.numel() > 0:
        meta.audio_sample_rate = float(asample_rate.item())
    return meta