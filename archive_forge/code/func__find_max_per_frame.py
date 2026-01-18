import math
import tempfile
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union
import torch
import torchaudio
from torch import Tensor
from torchaudio._internal.module_utils import deprecated
from .filtering import highpass_biquad, treble_biquad
def _find_max_per_frame(nccf: Tensor, sample_rate: int, freq_high: int) -> Tensor:
    """
    For each frame, take the highest value of NCCF,
    apply centered median smoothing, and convert to frequency.

    Note: If the max among all the lags is very close
    to the first half of lags, then the latter is taken.
    """
    lag_min = int(math.ceil(sample_rate / freq_high))
    best = torch.max(nccf[..., lag_min:], -1)
    half_size = nccf.shape[-1] // 2
    half = torch.max(nccf[..., lag_min:half_size], -1)
    best = _combine_max(half, best)
    indices = best[1]
    indices += lag_min
    indices += 1
    return indices