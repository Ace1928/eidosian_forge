import warnings
import numpy as np
from scipy import ndimage as ndi
from .footprints import _footprint_is_sequence, mirror_footprint, pad_footprint
from .misc import default_footprint
from .._shared.utils import DEPRECATED
def _shift_footprint(footprint, shift_x, shift_y):
    """Shift the binary image `footprint` in the left and/or up.

    This only affects 2D footprints with even number of rows
    or columns.

    Parameters
    ----------
    footprint : 2D array, shape (M, N)
        The input footprint.
    shift_x, shift_y : bool or None
        Whether to move `footprint` along each axis. If ``None``, the
        array is not modified along that dimension.

    Returns
    -------
    out : 2D array, shape (M + int(shift_x), N + int(shift_y))
        The shifted footprint.
    """
    footprint = np.asarray(footprint)
    if footprint.ndim != 2:
        return footprint
    m, n = footprint.shape
    if m % 2 == 0:
        extra_row = np.zeros((1, n), footprint.dtype)
        if shift_x:
            footprint = np.vstack((footprint, extra_row))
        else:
            footprint = np.vstack((extra_row, footprint))
        m += 1
    if n % 2 == 0:
        extra_col = np.zeros((m, 1), footprint.dtype)
        if shift_y:
            footprint = np.hstack((footprint, extra_col))
        else:
            footprint = np.hstack((extra_col, footprint))
    return footprint