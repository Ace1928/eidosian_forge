import numpy as np
import scipy.signal
from numba import jit
from .._cache import cache
from ..util.exceptions import ParameterError
from typing import Any
@jit(nopython=True, cache=True)
def __stack(history, data, n_steps, delay):
    """Memory-stacking helper function.

    Parameters
    ----------
    history : output array (2-dimensional)
    data : pre-padded input array (2-dimensional)
    n_steps : int > 0, the number of steps to stack
    delay : int != 0, the amount of delay between steps

    Returns
    -------
    None
        Output is stored directly in the history array
    """
    d = data.shape[-2]
    t = history.shape[-1]
    if delay > 0:
        for step in range(n_steps):
            q = n_steps - 1 - step
            history[..., step * d:(step + 1) * d, :] = data[..., q * delay:q * delay + t]
    else:
        history[..., -d:, :] = data[..., -t:]
        for step in range(n_steps - 1):
            q = n_steps - 1 - step
            history[..., step * d:(step + 1) * d, :] = data[..., -t + q * delay:q * delay]