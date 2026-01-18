from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
def _fft_func(self, x: np.ndarray) -> np.ndarray:
    """FFT based on the `fft_mode`, `mfft`, `scaling` and `phase_shift`
        attributes.

        For multidimensional arrays the transformation is carried out on the
        last axis.
        """
    if self.phase_shift is not None:
        if x.shape[-1] < self.mfft:
            z_shape = list(x.shape)
            z_shape[-1] = self.mfft - x.shape[-1]
            x = np.hstack((x, np.zeros(z_shape, dtype=x.dtype)))
        p_s = (self.phase_shift + self.m_num_mid) % self.m_num
        x = np.roll(x, -p_s, axis=-1)
    if self.fft_mode == 'twosided':
        return fft_lib.fft(x, n=self.mfft, axis=-1)
    if self.fft_mode == 'centered':
        return fft_lib.fftshift(fft_lib.fft(x, self.mfft, axis=-1))
    if self.fft_mode == 'onesided':
        return fft_lib.rfft(x, n=self.mfft, axis=-1)
    if self.fft_mode == 'onesided2X':
        X = fft_lib.rfft(x, n=self.mfft, axis=-1)
        fac = np.sqrt(2) if self.scaling == 'psd' else 2
        X[..., 1:-1 if self.mfft % 2 == 0 else None] *= fac
        return X
    fft_modes = get_args(FFT_MODE_TYPE)
    raise RuntimeError(f'self.fft_mode={self.fft_mode!r} not in {fft_modes}!')