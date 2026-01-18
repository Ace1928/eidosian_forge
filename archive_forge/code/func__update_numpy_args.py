import types
import numpy as np
import cupy as cp
from cupyx.fallback_mode import notification
def _update_numpy_args(args, kwargs):
    return _get_xp_args(ndarray, ndarray._update_numpy_array, (args, kwargs))