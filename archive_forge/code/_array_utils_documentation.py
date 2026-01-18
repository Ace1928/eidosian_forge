import cupy
from cupy.linalg import _util
Return the lower and upper bandwidth of a 2D numeric array.
    Parameters
    ----------
    a : ndarray
        Input array of size (M, N)
    Returns
    -------
    lu : tuple
        2-tuple of ints indicating the lower and upper bandwith. A zero
        denotes no sub- or super-diagonal on that side (triangular), and,
        say for M rows (M-1) means that side is full. Same example applies
        to the upper triangular part with (N-1).

    .. seealso:: :func:`scipy.linalg.bandwidth`
    