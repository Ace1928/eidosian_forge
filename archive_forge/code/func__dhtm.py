import math
from cupy.fft import fft, ifft
from cupy.linalg import solve, lstsq, LinAlgError
from cupyx.scipy.linalg import toeplitz, hankel
import cupyx
from cupyx.scipy.signal.windows import get_window
import cupy
import numpy
def _dhtm(mag):
    """Compute the modified 1-D discrete Hilbert transform

    Parameters
    ----------
    mag : ndarray
        The magnitude spectrum. Should be 1-D with an even length, and
        preferably a fast length for FFT/IFFT.
    """
    sig = cupy.zeros(len(mag))
    midpt = len(mag) // 2
    sig[1:midpt] = 1
    sig[midpt + 1:] = -1
    recon = ifft(mag * cupy.exp(fft(sig * ifft(cupy.log(mag))))).real
    return recon