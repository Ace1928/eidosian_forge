import math
import cupy
from cupy import _core
from cupyx.scipy.linalg import _uarray
Construct a convolution matrix.

    Constructs the Toeplitz matrix representing one-dimensional convolution.

    Args:
        a (cupy.ndarray): The 1-D array to convolve.
        n (int): The number of columns in the resulting matrix. It gives the
            length of the input to be convolved with ``a``. This is analogous
            to the length of ``v`` in ``numpy.convolve(a, v)``.
        mode (str): This must be one of (``'full'``, ``'valid'``, ``'same'``).
            This is analogous to ``mode`` in ``numpy.convolve(v, a, mode)``.

    Returns:
        cupy.ndarray: The convolution matrix whose row count k depends on
        ``mode``:

        =========== =========================
        ``mode``    k
        =========== =========================
        ``'full'``  m + n - 1
        ``'same'``  max(m, n)
        ``'valid'`` max(m, n) - min(m, n) + 1
        =========== =========================

    .. seealso:: :func:`cupyx.scipy.linalg.toeplitz`
    .. seealso:: :func:`scipy.linalg.convolution_matrix`
    