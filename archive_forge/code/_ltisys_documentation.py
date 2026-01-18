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

        Returns the discretized `StateSpace` system.

        Parameters: See `cont2discrete` for details.

        Returns
        -------
        sys: instance of `dlti` and `StateSpace`
        