import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm
from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
def _smw_logdet(s, A, AtA, Qi, di, B_logdet):
    """
    Returns the log determinant of

    .. math::

        sI + ABA^\\prime

    Uses the matrix determinant lemma to accelerate the calculation.
    B is assumed to be positive definite, and s > 0, therefore the
    determinant is positive.

    Parameters
    ----------
    s : positive scalar
        See above for usage
    A : ndarray
        p x q matrix, in general q << p.
    AtA : square ndarray
        :math:`A^\\prime  A`, a q x q matrix.
    Qi : square symmetric ndarray
        The matrix `B` is q x q, where q = r + d.  `B` consists of a r
        x r diagonal block whose inverse is `Qi`, and a d x d diagonal
        block, whose inverse is diag(di).
    di : 1d array_like
        See documentation for Qi.
    B_logdet : real
        The log determinant of B

    Returns
    -------
    The log determinant of s*I + A*B*A'.

    Notes
    -----
    Uses the matrix determinant lemma:
        https://en.wikipedia.org/wiki/Matrix_determinant_lemma
    """
    p = A.shape[0]
    ld = p * np.log(s)
    qmat = AtA / s
    m = Qi.shape[0]
    qmat[0:m, 0:m] += Qi
    if sparse.issparse(qmat):
        qmat[m:, m:] += sparse.diags(di)
        lu = sparse.linalg.splu(qmat)
        dl = lu.L.diagonal().astype(np.complex128)
        du = lu.U.diagonal().astype(np.complex128)
        ld1 = np.log(dl).sum() + np.log(du).sum()
        ld1 = ld1.real
    else:
        d = qmat.shape[0]
        qmat.flat[m * (d + 1)::d + 1] += di
        _, ld1 = np.linalg.slogdet(qmat)
    return B_logdet + ld + ld1