import warnings
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import brentq
from scipy.signal import convolve, convolve2d
from scipy.signal.windows import gaussian
from scipy.sparse import coo_matrix
from scipy.special import ive  # pylint: disable=no-name-in-module
from ..utils import _cov, _dot, _stack, conditional_jit
def _get_bw(x, bw, grid_counts=None, x_std=None, x_range=None):
    """Compute bandwidth for a given data `x` and `bw`.

    Also checks `bw` is correctly specified.

    Parameters
    ----------
    x : 1-D numpy array
        1 dimensional array of sample data from the
        variable for which a density estimate is desired.
    bw: int, float or str
        If numeric, indicates the bandwidth and must be positive.
        If str, indicates the method to estimate the bandwidth.

    Returns
    -------
    bw: float
        Bandwidth
    """
    if isinstance(bw, bool):
        raise ValueError(f'`bw` must not be of type `bool`.\nExpected a positive numeric or one of the following strings:\n{list(_BW_METHODS_LINEAR)}.')
    if isinstance(bw, (int, float)):
        if bw < 0:
            raise ValueError(f'Numeric `bw` must be positive.\nInput: {bw:.4f}.')
    elif isinstance(bw, str):
        bw_lower = bw.lower()
        if bw_lower not in _BW_METHODS_LINEAR:
            raise ValueError(f'Unrecognized bandwidth method.\nInput is: {bw_lower}.\nExpected one of: {list(_BW_METHODS_LINEAR)}.')
        bw_fun = _BW_METHODS_LINEAR[bw_lower]
        bw = bw_fun(x, grid_counts=grid_counts, x_std=x_std, x_range=x_range)
    else:
        raise ValueError(f'Unrecognized `bw` argument.\nExpected a positive numeric or one of the following strings:\n{list(_BW_METHODS_LINEAR)}.')
    return bw