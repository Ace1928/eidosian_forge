from warnings import warn
import numpy as np
import scipy.ndimage as ndi
from .. import measure
from .._shared.coord import ensure_spacing
def _get_threshold(image, threshold_abs, threshold_rel):
    """Return the threshold value according to an absolute and a relative
    value.

    """
    threshold = threshold_abs if threshold_abs is not None else image.min()
    if threshold_rel is not None:
        threshold = max(threshold, threshold_rel * image.max())
    return threshold