import math
from typing import Optional, Tuple, Union
import torch
import torchaudio
from torch import Tensor
def _frac_delay(delay: torch.Tensor, delay_i: torch.Tensor, delay_filter_length: int):
    """Compute fractional delay of impulse response signal.

    Args:
        delay (torch.Tensor): The time delay Tensor in samples.
        delay_i (torch.Tensor): The integer part of delay.
        delay_filter_length (int): The window length for sinc function.

    Returns:
        (torch.Tensor): The impulse response Tensor for all image sources.
    """
    if delay_filter_length % 2 != 1:
        raise ValueError('The filter length must be odd')
    pad = delay_filter_length // 2
    n = torch.arange(-pad, pad + 1, device=delay.device) + delay_i[..., None]
    delay = delay[..., None]
    return torch.special.sinc(n - delay) * _hann(n - delay, 2 * pad)