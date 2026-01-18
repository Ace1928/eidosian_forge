import math
import numpy as np
import scipy.linalg
from scipy._lib import doccer
from scipy.special import (gammaln, psi, multigammaln, xlogy, entr, betaln,
from scipy._lib._util import check_random_state, _lazywhere
from scipy.linalg.blas import drot, get_blas_funcs
from ._continuous_distns import norm
from ._discrete_distns import binom
from . import _mvn, _covariance, _rcont
from ._qmvnt import _qmvt
from ._morestats import directional_stats
from scipy.optimize import root_scalar
def _pinv_1d(v, eps=1e-05):
    """A helper function for computing the pseudoinverse.

    Parameters
    ----------
    v : iterable of numbers
        This may be thought of as a vector of eigenvalues or singular values.
    eps : float
        Values with magnitude no greater than eps are considered negligible.

    Returns
    -------
    v_pinv : 1d float ndarray
        A vector of pseudo-inverted numbers.

    """
    return np.array([0 if abs(x) <= eps else 1 / x for x in v], dtype=float)