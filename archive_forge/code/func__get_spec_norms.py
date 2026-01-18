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
def _get_spec_norms(normalized: Union[str, bool]):
    frame_length_norm, window_norm = (False, False)
    if torch.jit.isinstance(normalized, str):
        if normalized not in ['frame_length', 'window']:
            raise ValueError('Invalid normalized parameter: {}'.format(normalized))
        if normalized == 'frame_length':
            frame_length_norm = True
        elif normalized == 'window':
            window_norm = True
    elif torch.jit.isinstance(normalized, bool):
        if normalized:
            window_norm = True
    else:
        raise TypeError('Input type not supported')
    return (frame_length_norm, window_norm)