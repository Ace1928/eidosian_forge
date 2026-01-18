import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupy._core.internal import _normalize_axis_index
from cupyx.scipy.signal._signaltools import lfilter
from cupyx.scipy.signal._arraytools import (
from cupyx.scipy.signal._iir_utils import collapse_2d, apply_iir_sos

    Implement a smoothing IIR filter with mirror-symmetric boundary conditions
    using a cascade of second-order sections.  The second section uses a
    reversed sequence.  This implements the following transfer function::

                                  cs^2
         H(z) = ---------------------------------------
                (1 - a2/z - a3/z^2) (1 - a2 z - a3 z^2 )

    where::

          a2 = 2 * r * cos(omega)
          a3 = - r ** 2
          cs = 1 - 2 * r * cos(omega) + r ** 2

    Parameters
    ----------
    input : ndarray
        The input signal.
    r, omega : float
        Parameters in the transfer function.
    precision : float
        Specifies the precision for calculating initial conditions
        of the recursive filter based on mirror-symmetric input.

    Returns
    -------
    output : ndarray
        The filtered signal.
    