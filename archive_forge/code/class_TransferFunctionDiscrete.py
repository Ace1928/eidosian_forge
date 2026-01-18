import warnings
import copy
from math import sqrt
import cupy
from cupyx.scipy import linalg
from cupyx.scipy.interpolate import make_interp_spline
from cupyx.scipy.linalg import expm, block_diag
from cupyx.scipy.signal._lti_conversion import (
from cupyx.scipy.signal._iir_filter_conversions import (
from cupyx.scipy.signal._filter_design import (
class TransferFunctionDiscrete(TransferFunction, dlti):
    """
    Discrete-time Linear Time Invariant system in transfer function form.

    Represents the system as the transfer function
    :math:`H(z)=\\sum_{i=0}^N b[N-i] z^i / \\sum_{j=0}^M a[M-j] z^j`, where
    :math:`b` are elements of the numerator `num`, :math:`a` are elements of
    the denominator `den`, and ``N == len(b) - 1``, ``M == len(a) - 1``.
    Discrete-time `TransferFunction` systems inherit additional functionality
    from the `dlti` class.

    Parameters
    ----------
    *system: arguments
        The `TransferFunction` class can be instantiated with 1 or 2
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 2: array_like: (numerator, denominator)
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to `True`
        (unspecified sampling time). Must be specified as a keyword argument,
        for example, ``dt=0.1``.

    See Also
    --------
    scipy.signal.TransferFunctionDiscrete
    ZerosPolesGain, StateSpace, dlti
    tf2ss, tf2zpk, tf2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `TransferFunction` system representation (such as the `A`, `B`, `C`, `D`
    state-space matrices) is very inefficient and may lead to numerical
    inaccuracies.

    If (numerator, denominator) is passed in for ``*system``, coefficients
    for both the numerator and denominator should be specified in descending
    exponent order (e.g., ``z^2 + 3z + 5`` would be represented as
    ``[1, 3, 5]``).
    """
    pass