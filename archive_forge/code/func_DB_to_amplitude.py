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
def DB_to_amplitude(x: Tensor, ref: float, power: float) -> Tensor:
    """Turn a tensor from the decibel scale to the power/amplitude scale.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    Args:
        x (Tensor): Input tensor before being converted to power/amplitude scale.
        ref (float): Reference which the output will be scaled by.
        power (float): If power equals 1, will compute DB to power. If 0.5, will compute DB to amplitude.

    Returns:
        Tensor: Output tensor in power/amplitude scale.
    """
    return ref * torch.pow(torch.pow(10.0, 0.1 * x), power)