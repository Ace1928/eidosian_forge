import math
from warnings import warn
import cupy
from cupyx.scipy.fft import _fft
from cupyx.scipy.special import loggamma, poch
def _fhtq(a, u, inverse=False):
    """Compute the biased fast Hankel transform.

    This is the basic FFTLog routine.
    """
    n = a.shape[-1]
    if cupy.isinf(u[0]) and (not inverse):
        warn('singular transform; consider changing the bias')
        u = u.copy()
        u[0] = 0
    elif u[0] == 0 and inverse:
        warn('singular inverse transform; consider changing the bias')
        u = u.copy()
        u[0] = cupy.inf
    A = _fft.rfft(a, axis=-1)
    if not inverse:
        A *= u
    else:
        A /= u.conj()
    A = _fft.irfft(A, n, axis=-1)
    A = A[..., ::-1]
    return A