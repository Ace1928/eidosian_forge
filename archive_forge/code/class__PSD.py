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
class _PSD:
    """
    Compute coordinated functions of a symmetric positive semidefinite matrix.

    This class addresses two issues.  Firstly it allows the pseudoinverse,
    the logarithm of the pseudo-determinant, and the rank of the matrix
    to be computed using one call to eigh instead of three.
    Secondly it allows these functions to be computed in a way
    that gives mutually compatible results.
    All of the functions are computed with a common understanding as to
    which of the eigenvalues are to be considered negligibly small.
    The functions are designed to coordinate with scipy.linalg.pinvh()
    but not necessarily with np.linalg.det() or with np.linalg.matrix_rank().

    Parameters
    ----------
    M : array_like
        Symmetric positive semidefinite matrix (2-D).
    cond, rcond : float, optional
        Cutoff for small eigenvalues.
        Singular values smaller than rcond * largest_eigenvalue are
        considered zero.
        If None or -1, suitable machine precision is used.
    lower : bool, optional
        Whether the pertinent array data is taken from the lower
        or upper triangle of M. (Default: lower)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite
        numbers. Disabling may give a performance gain, but may result
        in problems (crashes, non-termination) if the inputs do contain
        infinities or NaNs.
    allow_singular : bool, optional
        Whether to allow a singular matrix.  (Default: True)

    Notes
    -----
    The arguments are similar to those of scipy.linalg.pinvh().

    """

    def __init__(self, M, cond=None, rcond=None, lower=True, check_finite=True, allow_singular=True):
        self._M = np.asarray(M)
        s, u = scipy.linalg.eigh(M, lower=lower, check_finite=check_finite)
        eps = _eigvalsh_to_eps(s, cond, rcond)
        if np.min(s) < -eps:
            msg = 'The input matrix must be symmetric positive semidefinite.'
            raise ValueError(msg)
        d = s[s > eps]
        if len(d) < len(s) and (not allow_singular):
            msg = 'When `allow_singular is False`, the input matrix must be symmetric positive definite.'
            raise np.linalg.LinAlgError(msg)
        s_pinv = _pinv_1d(s, eps)
        U = np.multiply(u, np.sqrt(s_pinv))
        self.eps = 1000.0 * eps
        self.V = u[:, s <= eps]
        self.rank = len(d)
        self.U = U
        self.log_pdet = np.sum(np.log(d))
        self._pinv = None

    def _support_mask(self, x):
        """
        Check whether x lies in the support of the distribution.
        """
        residual = np.linalg.norm(x @ self.V, axis=-1)
        in_support = residual < self.eps
        return in_support

    @property
    def pinv(self):
        if self._pinv is None:
            self._pinv = np.dot(self.U, self.U.T)
        return self._pinv