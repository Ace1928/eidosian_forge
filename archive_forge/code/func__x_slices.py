from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
def _x_slices(self, x: np.ndarray, k_off: int, p0: int, p1: int, padding: PAD_TYPE) -> Generator[np.ndarray, None, None]:
    """Generate signal slices along last axis of `x`.

        This method is only used by `stft_detrend`. The parameters are
        described in `~ShortTimeFFT.stft`.
        """
    if padding not in (padding_types := get_args(PAD_TYPE)):
        raise ValueError(f'Parameter padding={padding!r} not in {padding_types}!')
    pad_kws: dict[str, dict] = {'zeros': dict(mode='constant', constant_values=(0, 0)), 'edge': dict(mode='edge'), 'even': dict(mode='reflect', reflect_type='even'), 'odd': dict(mode='reflect', reflect_type='odd')}
    n, n1 = (x.shape[-1], (p1 - p0) * self.hop)
    k0 = p0 * self.hop - self.m_num_mid + k_off
    k1 = k0 + n1 + self.m_num
    i0, i1 = (max(k0, 0), min(k1, n))
    pad_width = [(0, 0)] * (x.ndim - 1) + [(-min(k0, 0), max(k1 - n, 0))]
    x1 = np.pad(x[..., i0:i1], pad_width, **pad_kws[padding])
    for k_ in range(0, n1, self.hop):
        yield x1[..., k_:k_ + self.m_num]