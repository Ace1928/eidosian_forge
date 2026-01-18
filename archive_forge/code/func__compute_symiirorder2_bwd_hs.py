import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupy._core.internal import _normalize_axis_index
from cupyx.scipy.signal._signaltools import lfilter
from cupyx.scipy.signal._arraytools import (
from cupyx.scipy.signal._iir_utils import collapse_2d, apply_iir_sos
def _compute_symiirorder2_bwd_hs(k, cs, rsq, omega):
    cssq = cs * cs
    k = cupy.abs(k)
    rsupk = cupy.power(rsq, k / 2.0)
    if omega == 0.0:
        c0 = (1 + rsq) / ((1 - rsq) * (1 - rsq) * (1 - rsq)) * cssq
        gamma = (1 - rsq) / (1 + rsq)
        return c0 * rsupk * (1 + gamma * k)
    if omega == cupy.pi:
        c0 = (1 + rsq) / ((1 - rsq) * (1 - rsq) * (1 - rsq)) * cssq
        gamma = (1 - rsq) / (1 + rsq) * (1 - 2 * (k % 2))
        return c0 * rsupk * (1 + gamma * k)
    c0 = cssq * (1.0 + rsq) / (1.0 - rsq) / (1 - 2 * rsq * cupy.cos(2 * omega) + rsq * rsq)
    gamma = (1.0 - rsq) / (1.0 + rsq) / cupy.tan(omega)
    return c0 * rsupk * (cupy.cos(omega * k) + gamma * cupy.sin(omega * k))