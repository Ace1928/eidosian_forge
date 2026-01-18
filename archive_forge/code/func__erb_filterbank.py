from functools import lru_cache
from math import ceil, pi
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn.functional import pad
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import (
def _erb_filterbank(wave: Tensor, coefs: Tensor) -> Tensor:
    """Translated from gammatone package.

    Args:
        wave: shape [B, time]
        coefs: shape [N, 10]

    Returns:
        Tensor: shape [B, N, time]

    """
    from torchaudio.functional.filtering import lfilter
    num_batch, time = wave.shape
    wave = wave.to(dtype=coefs.dtype).reshape(num_batch, 1, time)
    wave = wave.expand(-1, coefs.shape[0], -1)
    gain = coefs[:, 9]
    as1 = coefs[:, (0, 1, 5)]
    as2 = coefs[:, (0, 2, 5)]
    as3 = coefs[:, (0, 3, 5)]
    as4 = coefs[:, (0, 4, 5)]
    bs = coefs[:, 6:9]
    y1 = lfilter(wave, bs, as1, batching=True)
    y2 = lfilter(y1, bs, as2, batching=True)
    y3 = lfilter(y2, bs, as3, batching=True)
    y4 = lfilter(y3, bs, as4, batching=True)
    return y4 / gain.reshape(1, -1, 1)