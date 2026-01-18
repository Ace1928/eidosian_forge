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
def _validate_image_histogram(image, hist, nbins=None, normalize=False):
    """Ensure that either image or hist were given, return valid histogram.

    If hist is given, image is ignored.

    Parameters
    ----------
    image : array or None
        Grayscale image.
    hist : array, 2-tuple of array, or None
        Histogram, either a 1D counts array, or an array of counts together
        with an array of bin centers.
    nbins : int, optional
        The number of bins with which to compute the histogram, if `hist` is
        None.
    normalize : bool
        If hist is not given, it will be computed by this function. This
        parameter determines whether the computed histogram is normalized
        (i.e. entries sum up to 1) or not.

    Returns
    -------
    counts : 1D array of float
        Each element is the number of pixels falling in each intensity bin.
    bin_centers : 1D array
        Each element is the value corresponding to the center of each intensity
        bin.

    Raises
    ------
    ValueError : if image and hist are both None
    """
    if image is None and hist is None:
        raise Exception('Either image or hist must be provided.')
    if hist is not None:
        if isinstance(hist, tuple | list):
            counts, bin_centers = hist
        else:
            counts = hist
            bin_centers = np.arange(counts.size)
        if counts[0] == 0 or counts[-1] == 0:
            cond = counts > 0
            start = np.argmax(cond)
            end = cond.size - np.argmax(cond[::-1])
            counts, bin_centers = (counts[start:end], bin_centers[start:end])
    else:
        counts, bin_centers = histogram(image.reshape(-1), nbins, source_range='image', normalize=normalize)
    return (counts.astype('float32', copy=False), bin_centers)