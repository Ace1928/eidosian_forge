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
def _assert_psd_matrices(psd_s: torch.Tensor, psd_n: torch.Tensor) -> None:
    """Assertion checks of the PSD matrices of target speech and noise.

    Args:
        psd_s (torch.Tensor): The complex-valued power spectral density (PSD) matrix of target speech.
            Tensor with dimensions `(..., freq, channel, channel)`.
        psd_n (torch.Tensor): The complex-valued power spectral density (PSD) matrix of noise.
            Tensor with dimensions `(..., freq, channel, channel)`.
    """
    if psd_s.ndim < 3 or psd_n.ndim < 3:
        raise ValueError(f'Expected at least 3D Tensor (..., freq, channel, channel) for psd_s and psd_n. Found {psd_s.shape} for psd_s and {psd_n.shape} for psd_n.')
    if not (psd_s.is_complex() and psd_n.is_complex()):
        raise TypeError(f'The type of psd_s and psd_n must be ``torch.cfloat`` or ``torch.cdouble``. Found {psd_s.dtype} for psd_s and {psd_n.dtype} for psd_n.')
    if psd_s.shape != psd_n.shape:
        raise ValueError(f'The dimensions of psd_s and psd_n should be the same. Found {psd_s.shape} and {psd_n.shape}.')
    if psd_s.shape[-1] != psd_s.shape[-2]:
        raise ValueError(f'The last two dimensions of psd_s should be the same. Found {psd_s.shape}.')