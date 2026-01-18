import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupy._core.internal import _normalize_axis_index
from cupyx.scipy.signal._signaltools import lfilter
from cupyx.scipy.signal._arraytools import (
from cupyx.scipy.signal._iir_utils import collapse_2d, apply_iir_sos
def _find_initial_cond(all_valid, cum_poly, n, off=0, axis=-1):
    indices = cupy.where(all_valid)[0] + 1 + off
    zi = cupy.nan
    if indices.size > 0:
        zi = cupy.where(indices[0] >= n, cupy.nan, axis_slice(cum_poly, indices[0] - 1 - off, indices[0] - off, axis=axis))
    return zi