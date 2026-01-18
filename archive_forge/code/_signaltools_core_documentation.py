import math
import cupy
from cupy._core import internal
from cupyx.scipy import fft
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.ndimage import _util

    Computes the optimal block size for the OA method given the overlap size.

    Computed as ``ceil(-overlap*W(-1/(2*e*overlap)))`` where ``W(z)`` is the
    Lambert W function solved as per ``scipy.special.lambertw(z, -1)`` with a
    fixed 4 iterations.

    Returned size should still be given to ``cupyx.scipy.fft.next_fast_len()``.
    