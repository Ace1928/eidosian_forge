import numpy as np
from scipy import special
from scipy.optimize import OptimizeResult
from scipy.optimize._zeros_py import (  # noqa: F401
def _get_pairs(k, h0, inclusive=False, dtype=np.float64):
    if len(_pair_cache.indices) <= k + 2 or h0 != _pair_cache.h0:
        _pair_cache(k, h0)
    xjc = _pair_cache.xjc
    wj = _pair_cache.wj
    indices = _pair_cache.indices
    start = 0 if inclusive else indices[k]
    end = indices[k + 1]
    return (xjc[start:end].astype(dtype), wj[start:end].astype(dtype))