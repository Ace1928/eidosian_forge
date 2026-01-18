import numpy as np
from .._shared.utils import _supported_float_type, _to_np_mode
Compute valid cross-correlation of `padded_array` and `kernel`.

    This function is *fast* when `kernel` is large with many zeros.

    See ``scipy.ndimage.correlate`` for a description of cross-correlation.

    Parameters
    ----------
    image : ndarray, dtype float, shape (M, N[, ...], P)
        The input array. If mode is 'valid', this array should already be
        padded, as a margin of the same shape as kernel will be stripped
        off.
    kernel : ndarray, dtype float, shape (Q, R[, ...], S)
        The kernel to be correlated. Must have the same number of
        dimensions as `padded_array`. For high performance, it should
        be sparse (few nonzero entries).
    mode : string, optional
        See `scipy.ndimage.correlate` for valid modes.
        Additionally, mode 'valid' is accepted, in which case no padding is
        applied and the result is the result for the smaller image for which
        the kernel is entirely inside the original data.

    Returns
    -------
    result : array of float, shape (M, N[, ...], P)
        The result of cross-correlating `image` with `kernel`. If mode
        'valid' is used, the resulting shape is (M-Q+1, N-R+1[, ...], P-S+1).
    