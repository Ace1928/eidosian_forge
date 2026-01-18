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
class ZerosPolesGain(LinearTimeInvariant):
    """
    Linear Time Invariant system class in zeros, poles, gain form.

    Represents the system as the continuous- or discrete-time transfer function
    :math:`H(s)=k \\prod_i (s - z[i]) / \\prod_j (s - p[j])`, where :math:`k` is
    the `gain`, :math:`z` are the `zeros` and :math:`p` are the `poles`.
    `ZerosPolesGain` systems inherit additional functionality from the `lti`,
    respectively the `dlti` classes, depending on which system representation
    is used.

    Parameters
    ----------
    *system : arguments
        The `ZerosPolesGain` class can be instantiated with 1 or 3
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` or `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 3: array_like: (zeros, poles, gain)
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to `None`
        (continuous-time). Must be specified as a keyword argument, for
        example, ``dt=0.1``.


    See Also
    --------
    scipy.signal.ZerosPolesGain
    TransferFunction, StateSpace, lti, dlti
    zpk2ss, zpk2tf, zpk2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `ZerosPolesGain` system representation (such as the `A`, `B`, `C`, `D`
    state-space matrices) is very inefficient and may lead to numerical
    inaccuracies.  It is better to convert to the specific system
    representation first. For example, call ``sys = sys.to_ss()`` before
    accessing/changing the A, B, C, D system matrices.
    """

    def __new__(cls, *system, **kwargs):
        """Handle object conversion if input is an instance of `lti`"""
        if len(system) == 1 and isinstance(system[0], LinearTimeInvariant):
            return system[0].to_zpk()
        if cls is ZerosPolesGain:
            if kwargs.get('dt') is None:
                return ZerosPolesGainContinuous.__new__(ZerosPolesGainContinuous, *system, **kwargs)
            else:
                return ZerosPolesGainDiscrete.__new__(ZerosPolesGainDiscrete, *system, **kwargs)
        return super().__new__(cls)

    def __init__(self, *system, **kwargs):
        """Initialize the zeros, poles, gain system."""
        if isinstance(system[0], LinearTimeInvariant):
            return
        super().__init__(**kwargs)
        self._zeros = None
        self._poles = None
        self._gain = None
        self.zeros, self.poles, self.gain = system

    def __repr__(self):
        """Return representation of the `ZerosPolesGain` system."""
        return '{}(\n{},\n{},\n{},\ndt: {}\n)'.format(self.__class__.__name__, repr(self.zeros), repr(self.poles), repr(self.gain), repr(self.dt))

    @property
    def zeros(self):
        """Zeros of the `ZerosPolesGain` system."""
        return self._zeros

    @zeros.setter
    def zeros(self, zeros):
        self._zeros = cupy.atleast_1d(zeros)
        if len(self.zeros.shape) > 1:
            self.outputs, self.inputs = self.zeros.shape
        else:
            self.outputs = 1
            self.inputs = 1

    @property
    def poles(self):
        """Poles of the `ZerosPolesGain` system."""
        return self._poles

    @poles.setter
    def poles(self, poles):
        self._poles = cupy.atleast_1d(poles)

    @property
    def gain(self):
        """Gain of the `ZerosPolesGain` system."""
        return self._gain

    @gain.setter
    def gain(self, gain):
        self._gain = gain

    def _copy(self, system):
        """
        Copy the parameters of another `ZerosPolesGain` system.

        Parameters
        ----------
        system : instance of `ZerosPolesGain`
            The zeros, poles gain system that is to be copied

        """
        self.poles = system.poles
        self.zeros = system.zeros
        self.gain = system.gain

    def to_tf(self):
        """
        Convert system representation to `TransferFunction`.

        Returns
        -------
        sys : instance of `TransferFunction`
            Transfer function of the current system

        """
        return TransferFunction(*zpk2tf(self.zeros, self.poles, self.gain), **self._dt_dict)

    def to_zpk(self):
        """
        Return a copy of the current 'ZerosPolesGain' system.

        Returns
        -------
        sys : instance of `ZerosPolesGain`
            The current system (copy)

        """
        return copy.deepcopy(self)

    def to_ss(self):
        """
        Convert system representation to `StateSpace`.

        Returns
        -------
        sys : instance of `StateSpace`
            State space model of the current system

        """
        return StateSpace(*zpk2ss(self.zeros, self.poles, self.gain), **self._dt_dict)