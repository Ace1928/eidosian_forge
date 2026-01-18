import warnings
import cupy
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.signal import _signaltools_core as _st_core
Median filter a 2-dimensional array.

    Apply a median filter to the `input` array using a local window-size given
    by `kernel_size` (must be odd). The array is zero-padded automatically.

    Args:
        input (cupy.ndarray): A 2-dimensional input array.
        kernel_size (int of list of ints of length 2): Gives the size of the
            median filter window in each dimension. Elements of `kernel_size`
            should be odd. If `kernel_size` is a scalar, then this scalar is
            used as the size in each dimension. Default is a kernel of size
            (3, 3).

    Returns:
        cupy.ndarray: An array the same size as input containing the median
        filtered result.

    See also
    --------
    .. seealso:: :func:`cupyx.scipy.ndimage.median_filter`
    .. seealso:: :func:`cupyx.scipy.signal.medfilt`
    .. seealso:: :func:`scipy.signal.medfilt2d`
    