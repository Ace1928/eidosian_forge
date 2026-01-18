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
def _standard_rvs(self, n, shape, dim, df, random_state):
    """
        Parameters
        ----------
        n : integer
            Number of variates to generate
        shape : iterable
            Shape of the variates to generate
        dim : int
            Dimension of the scale matrix
        df : int
            Degrees of freedom
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'rvs' instead.

        """
    n_tril = dim * (dim - 1) // 2
    covariances = random_state.normal(size=n * n_tril).reshape(shape + (n_tril,))
    variances = np.r_[[random_state.chisquare(df - (i + 1) + 1, size=n) ** 0.5 for i in range(dim)]].reshape((dim,) + shape[::-1]).T
    A = np.zeros(shape + (dim, dim))
    size_idx = tuple([slice(None, None, None)] * len(shape))
    tril_idx = np.tril_indices(dim, k=-1)
    A[size_idx + tril_idx] = covariances
    diag_idx = np.diag_indices(dim)
    A[size_idx + diag_idx] = variances
    return A