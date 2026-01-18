from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
def _calc_dual_canonical_window(win: np.ndarray, hop: int) -> np.ndarray:
    """Calculate canonical dual window for 1d window `win` and a time step
    of `hop` samples.

    A ``ValueError`` is raised, if the inversion fails.

    This is a separate function not a method, since it is also used in the
    class method ``ShortTimeFFT.from_dual()``.
    """
    if hop > len(win):
        raise ValueError(f'hop={hop!r} is larger than window length of {len(win)}' + ' => STFT not invertible!')
    if issubclass(win.dtype.type, np.integer):
        raise ValueError("Parameter 'win' cannot be of integer type, but " + f'win.dtype={win.dtype!r} => STFT not invertible!')
    w2 = win.real ** 2 + win.imag ** 2
    DD = w2.copy()
    for k_ in range(hop, len(win), hop):
        DD[k_:] += w2[:-k_]
        DD[:-k_] += w2[k_:]
    relative_resolution = np.finfo(win.dtype).resolution * max(DD)
    if not np.all(DD >= relative_resolution):
        raise ValueError('Short-time Fourier Transform not invertible!')
    return win / DD