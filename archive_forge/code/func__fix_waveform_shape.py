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
def _fix_waveform_shape(waveform_shift: Tensor, shape: List[int]) -> Tensor:
    """
    PitchShift helper function to process after resampling step to fix the shape back.

    Args:
        waveform_shift(Tensor): The waveform after stretch and resample
        shape (List[int]): The shape of initial waveform

    Returns:
        Tensor: The pitch-shifted audio waveform of shape `(..., time)`.
    """
    ori_len = shape[-1]
    shift_len = waveform_shift.size()[-1]
    if shift_len > ori_len:
        waveform_shift = waveform_shift[..., :ori_len]
    else:
        waveform_shift = torch.nn.functional.pad(waveform_shift, [0, ori_len - shift_len])
    waveform_shift = waveform_shift.view(shape[:-1] + waveform_shift.shape[-1:])
    return waveform_shift