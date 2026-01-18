import math
from functools import partial
import numpy as np
import param
from ..core import NdOverlay, Overlay
from ..element.chart import Area
from .resample import ResampleOperation1D
def _lttb(x, y, n_out):
    """
    Downsample the data using the LTTB algorithm (python implementation).

    Args:
        x (np.ndarray): The x-values of the data.
        y (np.ndarray): The y-values of the data.
        n_out (int): The number of output points.
    Returns:
        np.array: The indexes of the selected datapoints.
    """
    block_size = (y.shape[0] - 2) / (n_out - 2)
    offset = np.arange(start=1, stop=y.shape[0], step=block_size).astype(np.int64)
    sampled_x = np.empty(n_out, dtype=np.int64)
    sampled_x[0] = 0
    sampled_x[-1] = x.shape[0] - 1
    if x.dtype.kind == 'M':
        x = x.view(np.int64)
    if y.dtype.kind == 'M':
        y = y.view(np.int64)
    _lttb_inner(x, y, n_out, sampled_x, offset)
    return sampled_x