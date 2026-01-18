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
class matrix_normal_frozen(multi_rv_frozen):
    """
    Create a frozen matrix normal distribution.

    Parameters
    ----------
    %(_matnorm_doc_default_callparams)s
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is `None` the `~np.random.RandomState` singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used, seeded
        with seed.
        If `seed` is already a ``RandomState`` or ``Generator`` instance,
        then that object is used.
        Default is `None`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import matrix_normal

    >>> distn = matrix_normal(mean=np.zeros((3,3)))
    >>> X = distn.rvs(); X
    array([[-0.02976962,  0.93339138, -0.09663178],
           [ 0.67405524,  0.28250467, -0.93308929],
           [-0.31144782,  0.74535536,  1.30412916]])
    >>> distn.pdf(X)
    2.5160642368346784e-05
    >>> distn.logpdf(X)
    -10.590229595124615
    """

    def __init__(self, mean=None, rowcov=1, colcov=1, seed=None):
        self._dist = matrix_normal_gen(seed)
        self.dims, self.mean, self.rowcov, self.colcov = self._dist._process_parameters(mean, rowcov, colcov)
        self.rowpsd = _PSD(self.rowcov, allow_singular=False)
        self.colpsd = _PSD(self.colcov, allow_singular=False)

    def logpdf(self, X):
        X = self._dist._process_quantiles(X, self.dims)
        out = self._dist._logpdf(self.dims, X, self.mean, self.rowpsd.U, self.rowpsd.log_pdet, self.colpsd.U, self.colpsd.log_pdet)
        return _squeeze_output(out)

    def pdf(self, X):
        return np.exp(self.logpdf(X))

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.mean, self.rowcov, self.colcov, size, random_state)

    def entropy(self):
        return self._dist._entropy(self.dims, self.rowpsd.log_pdet, self.colpsd.log_pdet)