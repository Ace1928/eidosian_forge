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