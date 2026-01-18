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
class LinearTimeInvariant:

    def __new__(cls, *system, **kwargs):
        """Create a new object, don't allow direct instances."""
        if cls is LinearTimeInvariant:
            raise NotImplementedError('The LinearTimeInvariant class is not meant to be used directly, use `lti` or `dlti` instead.')
        return super().__new__(cls)

    def __init__(self):
        """
        Initialize the `lti` baseclass.

        The heavy lifting is done by the subclasses.
        """
        super().__init__()
        self.inputs = None
        self.outputs = None
        self._dt = None

    @property
    def dt(self):
        """Return the sampling time of the system, `None` for `lti` systems."""
        return self._dt

    @property
    def _dt_dict(self):
        if self.dt is None:
            return {}
        else:
            return {'dt': self.dt}

    @property
    def zeros(self):
        """Zeros of the system."""
        return self.to_zpk().zeros

    @property
    def poles(self):
        """Poles of the system."""
        return self.to_zpk().poles

    def _as_ss(self):
        """Convert to `StateSpace` system, without copying.

        Returns
        -------
        sys: StateSpace
            The `StateSpace` system. If the class is already an instance of
            `StateSpace` then this instance is returned.
        """
        if isinstance(self, StateSpace):
            return self
        else:
            return self.to_ss()

    def _as_zpk(self):
        """Convert to `ZerosPolesGain` system, without copying.

        Returns
        -------
        sys: ZerosPolesGain
            The `ZerosPolesGain` system. If the class is already an instance of
            `ZerosPolesGain` then this instance is returned.
        """
        if isinstance(self, ZerosPolesGain):
            return self
        else:
            return self.to_zpk()

    def _as_tf(self):
        """Convert to `TransferFunction` system, without copying.

        Returns
        -------
        sys: ZerosPolesGain
            The `TransferFunction` system. If the class is already an instance
            of `TransferFunction` then this instance is returned.
        """
        if isinstance(self, TransferFunction):
            return self
        else:
            return self.to_tf()