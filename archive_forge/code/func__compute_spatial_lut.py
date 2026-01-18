import functools
from math import ceil
import numbers
import scipy.stats
import numpy as np
from ..util.dtype import img_as_float
from .._shared import utils
from .._shared.utils import _supported_float_type, warn
from ._denoise_cy import _denoise_bilateral, _denoise_tv_bregman
from .. import color
from ..color.colorconv import ycbcr_from_rgb
def _compute_spatial_lut(win_size, sigma, *, dtype=float):
    """Helping function. Define a lookup table containing Gaussian filter
    values using the spatial sigma.

    Parameters
    ----------
    win_size : int
        Window size for filtering.
        If win_size is not specified, it is calculated as
        ``max(5, 2 * ceil(3 * sigma_spatial) + 1)``.
    sigma : float
        Standard deviation for range distance. A larger value results in
        averaging of pixels with larger spatial differences.
    dtype : data type object
        The type and size of the data to be returned.

    Returns
    -------
    spatial_lut : ndarray
        Lookup table for the spatial sigma.
    """
    grid_points = np.arange(-win_size // 2, win_size // 2 + 1)
    rr, cc = np.meshgrid(grid_points, grid_points, indexing='ij')
    distances = np.hypot(rr, cc)
    return _gaussian_weight(distances, sigma ** 2, dtype=dtype).ravel()