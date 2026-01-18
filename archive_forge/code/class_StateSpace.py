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
class StateSpace(LinearTimeInvariant):
    """
    Linear Time Invariant system in state-space form.

    Represents the system as the continuous-time, first order differential
    equation :math:`\\dot{x} = A x + B u` or the discrete-time difference
    equation :math:`x[k+1] = A x[k] + B u[k]`. `StateSpace` systems
    inherit additional functionality from the `lti`, respectively the `dlti`
    classes, depending on which system representation is used.

    Parameters
    ----------
    *system: arguments
        The `StateSpace` class can be instantiated with 1 or 4 arguments.
        The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` or `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 4: array_like: (A, B, C, D)
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to `None`
        (continuous-time). Must be specified as a keyword argument, for
        example, ``dt=0.1``.

    See Also
    --------
    scipy.signal.StateSpace
    TransferFunction, ZerosPolesGain, lti, dlti
    ss2zpk, ss2tf, zpk2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `StateSpace` system representation (such as `zeros` or `poles`) is very
    inefficient and may lead to numerical inaccuracies.  It is better to
    convert to the specific system representation first. For example, call
    ``sys = sys.to_zpk()`` before accessing/changing the zeros, poles or gain.
    """
    __array_priority__ = 100.0
    __array_ufunc__ = None

    def __new__(cls, *system, **kwargs):
        """Create new StateSpace object and settle inheritance."""
        if len(system) == 1 and isinstance(system[0], LinearTimeInvariant):
            return system[0].to_ss()
        if cls is StateSpace:
            if kwargs.get('dt') is None:
                return StateSpaceContinuous.__new__(StateSpaceContinuous, *system, **kwargs)
            else:
                return StateSpaceDiscrete.__new__(StateSpaceDiscrete, *system, **kwargs)
        return super().__new__(cls)

    def __init__(self, *system, **kwargs):
        """Initialize the state space lti/dlti system."""
        if isinstance(system[0], LinearTimeInvariant):
            return
        super().__init__(**kwargs)
        self._A = None
        self._B = None
        self._C = None
        self._D = None
        self.A, self.B, self.C, self.D = abcd_normalize(*system)

    def __repr__(self):
        """Return representation of the `StateSpace` system."""
        return '{}(\n{},\n{},\n{},\n{},\ndt: {}\n)'.format(self.__class__.__name__, repr(self.A), repr(self.B), repr(self.C), repr(self.D), repr(self.dt))

    def _check_binop_other(self, other):
        return isinstance(other, (StateSpace, cupy.ndarray, float, complex, cupy.number, int))

    def __mul__(self, other):
        """
        Post-multiply another system or a scalar

        Handles multiplication of systems in the sense of a frequency domain
        multiplication. That means, given two systems E1(s) and E2(s), their
        multiplication, H(s) = E1(s) * E2(s), means that applying H(s) to U(s)
        is equivalent to first applying E2(s), and then E1(s).

        Notes
        -----
        For SISO systems the order of system application does not matter.
        However, for MIMO systems, where the two systems are matrices, the
        order above ensures standard Matrix multiplication rules apply.
        """
        if not self._check_binop_other(other):
            return NotImplemented
        if isinstance(other, StateSpace):
            if type(other) is not type(self):
                return NotImplemented
            if self.dt != other.dt:
                raise TypeError('Cannot multiply systems with different `dt`.')
            n1 = self.A.shape[0]
            n2 = other.A.shape[0]
            a = cupy.vstack((cupy.hstack((self.A, self.B @ other.C)), cupy.hstack((cupy.zeros((n2, n1)), other.A))))
            b = cupy.vstack((self.B @ other.D, other.B))
            c = cupy.hstack((self.C, self.D @ other.C))
            d = self.D @ other.D
        else:
            a = self.A
            b = self.B @ other
            c = self.C
            d = self.D @ other
        common_dtype = cupy.result_type(a.dtype, b.dtype, c.dtype, d.dtype)
        return StateSpace(cupy.asarray(a, dtype=common_dtype), cupy.asarray(b, dtype=common_dtype), cupy.asarray(c, dtype=common_dtype), cupy.asarray(d, dtype=common_dtype), **self._dt_dict)

    def __rmul__(self, other):
        """Pre-multiply a scalar or matrix (but not StateSpace)"""
        if not self._check_binop_other(other) or isinstance(other, StateSpace):
            return NotImplemented
        a = self.A
        b = self.B
        c = other @ self.C
        d = other @ self.D
        common_dtype = cupy.result_type(a.dtype, b.dtype, c.dtype, d.dtype)
        return StateSpace(cupy.asarray(a, dtype=common_dtype), cupy.asarray(b, dtype=common_dtype), cupy.asarray(c, dtype=common_dtype), cupy.asarray(d, dtype=common_dtype), **self._dt_dict)

    def __neg__(self):
        """Negate the system (equivalent to pre-multiplying by -1)."""
        return StateSpace(self.A, self.B, -self.C, -self.D, **self._dt_dict)

    def __add__(self, other):
        """
        Adds two systems in the sense of frequency domain addition.
        """
        if not self._check_binop_other(other):
            return NotImplemented
        if isinstance(other, StateSpace):
            if type(other) is not type(self):
                raise TypeError('Cannot add {} and {}'.format(type(self), type(other)))
            if self.dt != other.dt:
                raise TypeError('Cannot add systems with different `dt`.')
            a = block_diag(self.A, other.A)
            b = cupy.vstack((self.B, other.B))
            c = cupy.hstack((self.C, other.C))
            d = self.D + other.D
        else:
            other = cupy.atleast_2d(other)
            if self.D.shape == other.shape:
                a = self.A
                b = self.B
                c = self.C
                d = self.D + other
            else:
                raise ValueError('Cannot add systems with incompatible dimensions ({} and {})'.format(self.D.shape, other.shape))
        common_dtype = cupy.result_type(a.dtype, b.dtype, c.dtype, d.dtype)
        return StateSpace(cupy.asarray(a, dtype=common_dtype), cupy.asarray(b, dtype=common_dtype), cupy.asarray(c, dtype=common_dtype), cupy.asarray(d, dtype=common_dtype), **self._dt_dict)

    def __sub__(self, other):
        if not self._check_binop_other(other):
            return NotImplemented
        return self.__add__(-other)

    def __radd__(self, other):
        if not self._check_binop_other(other):
            return NotImplemented
        return self.__add__(other)

    def __rsub__(self, other):
        if not self._check_binop_other(other):
            return NotImplemented
        return (-self).__add__(other)

    def __truediv__(self, other):
        """
        Divide by a scalar
        """
        if not self._check_binop_other(other) or isinstance(other, StateSpace):
            return NotImplemented
        if isinstance(other, cupy.ndarray) and other.ndim > 0:
            raise ValueError('Cannot divide StateSpace by non-scalar numpy arrays')
        return self.__mul__(1 / other)

    @property
    def A(self):
        """State matrix of the `StateSpace` system."""
        return self._A

    @A.setter
    def A(self, A):
        self._A = _atleast_2d_or_none(A)

    @property
    def B(self):
        """Input matrix of the `StateSpace` system."""
        return self._B

    @B.setter
    def B(self, B):
        self._B = _atleast_2d_or_none(B)
        self.inputs = self.B.shape[-1]

    @property
    def C(self):
        """Output matrix of the `StateSpace` system."""
        return self._C

    @C.setter
    def C(self, C):
        self._C = _atleast_2d_or_none(C)
        self.outputs = self.C.shape[0]

    @property
    def D(self):
        """Feedthrough matrix of the `StateSpace` system."""
        return self._D

    @D.setter
    def D(self, D):
        self._D = _atleast_2d_or_none(D)

    def _copy(self, system):
        """
        Copy the parameters of another `StateSpace` system.

        Parameters
        ----------
        system : instance of `StateSpace`
            The state-space system that is to be copied

        """
        self.A = system.A
        self.B = system.B
        self.C = system.C
        self.D = system.D

    def to_tf(self, **kwargs):
        """
        Convert system representation to `TransferFunction`.

        Parameters
        ----------
        kwargs : dict, optional
            Additional keywords passed to `ss2zpk`

        Returns
        -------
        sys : instance of `TransferFunction`
            Transfer function of the current system

        """
        return TransferFunction(*ss2tf(self._A, self._B, self._C, self._D, **kwargs), **self._dt_dict)

    def to_zpk(self, **kwargs):
        """
        Convert system representation to `ZerosPolesGain`.

        Parameters
        ----------
        kwargs : dict, optional
            Additional keywords passed to `ss2zpk`

        Returns
        -------
        sys : instance of `ZerosPolesGain`
            Zeros, poles, gain representation of the current system

        """
        return ZerosPolesGain(*ss2zpk(self._A, self._B, self._C, self._D, **kwargs), **self._dt_dict)

    def to_ss(self):
        """
        Return a copy of the current `StateSpace` system.

        Returns
        -------
        sys : instance of `StateSpace`
            The current system (copy)

        """
        return copy.deepcopy(self)