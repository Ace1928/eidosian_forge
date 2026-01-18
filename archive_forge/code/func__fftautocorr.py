from CuSignal under terms of the MIT license.
import warnings
from typing import Set
import cupy
import numpy as np
def _fftautocorr(x):
    """Compute the autocorrelation of a real array and crop the result."""
    N = x.shape[-1]
    use_N = cupy.fft.next_fast_len(2 * N - 1)
    x_fft = cupy.fft.rfft(x, use_N, axis=-1)
    cxy = cupy.fft.irfft(x_fft * x_fft.conj(), n=use_N)[:, :N]
    return cxy