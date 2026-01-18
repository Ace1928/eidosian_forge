import types
import numpy as np
import cupy as cp
from cupyx.fallback_mode import notification
def _convert_cupy_to_fallback(cupy_res):
    return _get_xp_args(cp.ndarray, ndarray._store_array_from_cupy, cupy_res)