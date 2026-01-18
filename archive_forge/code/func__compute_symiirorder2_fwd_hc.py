import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupy._core.internal import _normalize_axis_index
from cupyx.scipy.signal._signaltools import lfilter
from cupyx.scipy.signal._arraytools import (
from cupyx.scipy.signal._iir_utils import collapse_2d, apply_iir_sos
def _compute_symiirorder2_fwd_hc(k, cs, r, omega):
    base = None
    if omega == 0.0:
        base = cs * cupy.power(r, k) * (k + 1)
    elif omega == cupy.pi:
        base = cs * cupy.power(r, k) * (k + 1) * (1 - 2 * (k % 2))
    else:
        base = cs * cupy.power(r, k) * cupy.sin(omega * (k + 1)) / cupy.sin(omega)
    return cupy.where(k < 0, 0.0, base)