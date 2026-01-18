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
class ZerosPolesGainContinuous(ZerosPolesGain, lti):
    """
    Continuous-time Linear Time Invariant system in zeros, poles, gain form.

    Represents the system as the continuous time transfer function
    :math:`H(s)=k \\prod_i (s - z[i]) / \\prod_j (s - p[j])`, where :math:`k` is
    the `gain`, :math:`z` are the `zeros` and :math:`p` are the `poles`.
    Continuous-time `ZerosPolesGain` systems inherit additional functionality
    from the `lti` class.

    Parameters
    ----------
    *system : arguments
        The `ZerosPolesGain` class can be instantiated with 1 or 3
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 3: array_like: (zeros, poles, gain)

    See Also
    --------
    TransferFunction, StateSpace, lti
    zpk2ss, zpk2tf, zpk2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `ZerosPolesGain` system representation (such as the `A`, `B`, `C`, `D`
    state-space matrices) is very inefficient and may lead to numerical
    inaccuracies.  It is better to convert to the specific system
    representation first. For example, call ``sys = sys.to_ss()`` before
    accessing/changing the A, B, C, D system matrices.

    Examples
    --------
    Construct the transfer function
    :math:`H(s)=\\frac{5(s - 1)(s - 2)}{(s - 3)(s - 4)}`:

    >>> from scipy import signal

    >>> signal.ZerosPolesGain([1, 2], [3, 4], 5)
    ZerosPolesGainContinuous(
    array([1, 2]),
    array([3, 4]),
    5,
    dt: None
    )

    """

    def to_discrete(self, dt, method='zoh', alpha=None):
        """
        Returns the discretized `ZerosPolesGain` system.

        Parameters: See `cont2discrete` for details.

        Returns
        -------
        sys: instance of `dlti` and `ZerosPolesGain`
        """
        return ZerosPolesGain(*cont2discrete((self.zeros, self.poles, self.gain), dt, method=method, alpha=alpha)[:-1], dt=dt)