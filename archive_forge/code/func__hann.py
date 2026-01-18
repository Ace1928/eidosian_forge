import math
from typing import Optional, Tuple, Union
import torch
import torchaudio
from torch import Tensor
def _hann(x: torch.Tensor, T: int):
    """Compute the Hann window where the values are truncated based on window length.
    torch.hann_window can only sample window function at integer points, the method is to sample
    continuous window function at non-integer points.

    Args:
        x (torch.Tensor): The fractional component of time delay Tensor.
        T (torch.Tensor): The window length of sinc function.

    Returns:
        (torch.Tensor): The hann window Tensor where values outside
            the sinc window (`T`) is set to zero.
    """
    y = torch.where(torch.abs(x) <= T / 2, 0.5 * (1 + torch.cos(2 * math.pi * x / T)), x.new_zeros(1))
    return y