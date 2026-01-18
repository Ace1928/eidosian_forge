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
def _inv_standard_rvs(self, n, shape, dim, df, random_state):
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

        Returns
        -------
        A : ndarray
            Random variates of shape (`shape`) + (``dim``, ``dim``).
            Each slice `A[..., :, :]` is lower-triangular, and its
            inverse is the lower Cholesky factor of a draw from
            `invwishart(df, np.eye(dim))`.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'rvs' instead.

        """
    A = np.zeros(shape + (dim, dim))
    tri_rows, tri_cols = np.tril_indices(dim, k=-1)
    n_tril = dim * (dim - 1) // 2
    A[..., tri_rows, tri_cols] = random_state.normal(size=(*shape, n_tril))
    rows = np.arange(dim)
    chi_dfs = df - dim + 1 + rows
    A[..., rows, rows] = random_state.chisquare(df=chi_dfs, size=(*shape, dim)) ** 0.5
    return A