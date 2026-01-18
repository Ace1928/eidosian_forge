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
def _cholesky_logdet(self, scale):
    """Compute Cholesky decomposition and determine (log(det(scale)).

        Parameters
        ----------
        scale : ndarray
            Scale matrix.

        Returns
        -------
        c_decomp : ndarray
            The Cholesky decomposition of `scale`.
        logdet : scalar
            The log of the determinant of `scale`.

        Notes
        -----
        This computation of ``logdet`` is equivalent to
        ``np.linalg.slogdet(scale)``.  It is ~2x faster though.

        """
    c_decomp = scipy.linalg.cholesky(scale, lower=True)
    logdet = 2 * np.sum(np.log(c_decomp.diagonal()))
    return (c_decomp, logdet)