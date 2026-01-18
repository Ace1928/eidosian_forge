import bisect
import math
import warnings
from fractions import Fraction
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, TypeVar, Union
import torch
from torchvision.io import _probe_video_from_file, _read_video_from_file, read_video, read_video_timestamps
from .utils import tqdm
@staticmethod
def _resample_video_idx(num_frames: int, original_fps: int, new_fps: int) -> Union[slice, torch.Tensor]:
    step = float(original_fps) / new_fps
    if step.is_integer():
        step = int(step)
        return slice(None, None, step)
    idxs = torch.arange(num_frames, dtype=torch.float32) * step
    idxs = idxs.floor().to(torch.int64)
    return idxs