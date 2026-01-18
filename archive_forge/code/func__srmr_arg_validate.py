from functools import lru_cache
from math import ceil, pi
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn.functional import pad
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import (
def _srmr_arg_validate(fs: int, n_cochlear_filters: int=23, low_freq: float=125, min_cf: float=4, max_cf: Optional[float]=128, norm: bool=False, fast: bool=False) -> None:
    """Validate the arguments for speech_reverberation_modulation_energy_ratio.

    Args:
        fs: the sampling rate
        n_cochlear_filters: Number of filters in the acoustic filterbank
        low_freq: determines the frequency cutoff for the corresponding gammatone filterbank.
        min_cf: Center frequency in Hz of the first modulation filter.
        max_cf: Center frequency in Hz of the last modulation filter. If None is given,
        norm: Use modulation spectrum energy normalization
        fast: Use the faster version based on the gammatonegram.

    """
    if not (isinstance(fs, int) and fs > 0):
        raise ValueError(f'Expected argument `fs` to be an int larger than 0, but got {fs}')
    if not (isinstance(n_cochlear_filters, int) and n_cochlear_filters > 0):
        raise ValueError(f'Expected argument `n_cochlear_filters` to be an int larger than 0, but got {n_cochlear_filters}')
    if not (isinstance(low_freq, (float, int)) and low_freq > 0):
        raise ValueError(f'Expected argument `low_freq` to be a float larger than 0, but got {low_freq}')
    if not (isinstance(min_cf, (float, int)) and min_cf > 0):
        raise ValueError(f'Expected argument `min_cf` to be a float larger than 0, but got {min_cf}')
    if max_cf is not None and (not (isinstance(max_cf, (float, int)) and max_cf > 0)):
        raise ValueError(f'Expected argument `max_cf` to be a float larger than 0, but got {max_cf}')
    if not isinstance(norm, bool):
        raise ValueError('Expected argument `norm` to be a bool value')
    if not isinstance(fast, bool):
        raise ValueError('Expected argument `fast` to be a bool value')