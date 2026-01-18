import math
import typing as tp
from typing import Any, Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
def _ispectro(z: torch.Tensor, hop_length: int=0, length: int=0, pad: int=0) -> torch.Tensor:
    other = list(z.shape[:-2])
    freqs = int(z.shape[-2])
    frames = int(z.shape[-1])
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    x = torch.istft(z, n_fft, hop_length, window=torch.hann_window(win_length).to(z.real), win_length=win_length, normalized=True, length=length, center=True)
    _, length = x.shape
    other.append(length)
    return x.view(other)