from functools import lru_cache
from math import ceil, pi
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn.functional import pad
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import (
@lru_cache(maxsize=100)
def _compute_modulation_filterbank_and_cutoffs(min_cf: float, max_cf: float, n: int, fs: float, q: int, device: torch.device) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    spacing_factor = (max_cf / min_cf) ** (1.0 / (n - 1))
    cfs = torch.zeros(n, dtype=torch.float64)
    cfs[0] = min_cf
    for k in range(1, n):
        cfs[k] = cfs[k - 1] * spacing_factor

    def _make_modulation_filter(w0: Tensor, q: int) -> Tensor:
        w0 = torch.tan(w0 / 2)
        b0 = w0 / q
        b = torch.tensor([b0, 0, -b0], dtype=torch.float64)
        a = torch.tensor([1 + b0 + w0 ** 2, 2 * w0 ** 2 - 2, 1 - b0 + w0 ** 2], dtype=torch.float64)
        return torch.stack([b, a], dim=0)
    mfb = torch.stack([_make_modulation_filter(w0, q) for w0 in 2 * pi * cfs / fs], dim=0)

    def _calc_cutoffs(cfs: Tensor, fs: float, q: int) -> Tuple[Tensor, Tensor]:
        w0 = 2 * pi * cfs / fs
        b0 = torch.tan(w0 / 2) / q
        ll = cfs - b0 * fs / (2 * pi)
        rr = cfs + b0 * fs / (2 * pi)
        return (ll, rr)
    cfs = cfs.to(device=device)
    mfb = mfb.to(device=device)
    ll, rr = _calc_cutoffs(cfs, fs, q)
    return (cfs, mfb, ll, rr)