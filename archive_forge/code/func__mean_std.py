import inspect
import itertools
import math
from collections import OrderedDict
from collections.abc import Iterable
import numpy as np
from scipy import ndimage as ndi
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, warn
from .._shared.version_requirements import require
from ..exposure import histogram
from ..filters._multiotsu import (
from ..transform import integral_image
from ..util import dtype_limits
from ._sparse import _correlate_sparse, _validate_window_size
def _mean_std(image, w):
    """Return local mean and standard deviation of each pixel using a
    neighborhood defined by a rectangular window size ``w``.
    The algorithm uses integral images to speedup computation. This is
    used by :func:`threshold_niblack` and :func:`threshold_sauvola`.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    w : int, or iterable of int
        Window size specified as a single odd integer (3, 5, 7, …),
        or an iterable of length ``image.ndim`` containing only odd
        integers (e.g. ``(1, 5, 5)``).

    Returns
    -------
    m : ndarray of float, same shape as ``image``
        Local mean of the image.
    s : ndarray of float, same shape as ``image``
        Local standard deviation of the image.

    References
    ----------
    .. [1] F. Shafait, D. Keysers, and T. M. Breuel, "Efficient
           implementation of local adaptive thresholding techniques
           using integral images." in Document Recognition and
           Retrieval XV, (San Jose, USA), Jan. 2008.
           :DOI:`10.1117/12.767755`
    """
    if not isinstance(w, Iterable):
        w = (w,) * image.ndim
    _validate_window_size(w)
    float_dtype = _supported_float_type(image.dtype)
    pad_width = tuple(((k // 2 + 1, k // 2) for k in w))
    padded = np.pad(image.astype(float_dtype, copy=False), pad_width, mode='reflect')
    integral = integral_image(padded, dtype=np.float64)
    padded *= padded
    integral_sq = integral_image(padded, dtype=np.float64)
    kernel_indices = list(itertools.product(*tuple([(0, _w) for _w in w])))
    kernel_values = [(-1) ** (image.ndim % 2 != np.sum(indices) % 2) for indices in kernel_indices]
    total_window_size = math.prod(w)
    kernel_shape = tuple((_w + 1 for _w in w))
    m = _correlate_sparse(integral, kernel_shape, kernel_indices, kernel_values)
    m = m.astype(float_dtype, copy=False)
    m /= total_window_size
    g2 = _correlate_sparse(integral_sq, kernel_shape, kernel_indices, kernel_values)
    g2 = g2.astype(float_dtype, copy=False)
    g2 /= total_window_size
    s = np.sqrt(np.clip(g2 - m * m, 0, None))
    return (m, s)