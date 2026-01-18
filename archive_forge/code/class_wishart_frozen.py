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
class wishart_frozen(multi_rv_frozen):
    """Create a frozen Wishart distribution.

    Parameters
    ----------
    df : array_like
        Degrees of freedom of the distribution
    scale : array_like
        Scale matrix of the distribution
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    """

    def __init__(self, df, scale, seed=None):
        self._dist = wishart_gen(seed)
        self.dim, self.df, self.scale = self._dist._process_parameters(df, scale)
        self.C, self.log_det_scale = self._dist._cholesky_logdet(self.scale)

    def logpdf(self, x):
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._logpdf(x, self.dim, self.df, self.scale, self.log_det_scale, self.C)
        return _squeeze_output(out)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def mean(self):
        out = self._dist._mean(self.dim, self.df, self.scale)
        return _squeeze_output(out)

    def mode(self):
        out = self._dist._mode(self.dim, self.df, self.scale)
        return _squeeze_output(out) if out is not None else out

    def var(self):
        out = self._dist._var(self.dim, self.df, self.scale)
        return _squeeze_output(out)

    def rvs(self, size=1, random_state=None):
        n, shape = self._dist._process_size(size)
        out = self._dist._rvs(n, shape, self.dim, self.df, self.C, random_state)
        return _squeeze_output(out)

    def entropy(self):
        return self._dist._entropy(self.dim, self.df, self.log_det_scale)