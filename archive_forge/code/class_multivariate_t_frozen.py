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
class multivariate_t_frozen(multi_rv_frozen):

    def __init__(self, loc=None, shape=1, df=1, allow_singular=False, seed=None):
        """Create a frozen multivariate t distribution.

        Parameters
        ----------
        %(_mvt_doc_default_callparams)s

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import multivariate_t
        >>> loc = np.zeros(3)
        >>> shape = np.eye(3)
        >>> df = 10
        >>> dist = multivariate_t(loc, shape, df)
        >>> dist.rvs()
        array([[ 0.81412036, -1.53612361,  0.42199647]])
        >>> dist.pdf([1, 1, 1])
        array([0.01237803])

        """
        self._dist = multivariate_t_gen(seed)
        dim, loc, shape, df = self._dist._process_parameters(loc, shape, df)
        self.dim, self.loc, self.shape, self.df = (dim, loc, shape, df)
        self.shape_info = _PSD(shape, allow_singular=allow_singular)

    def logpdf(self, x):
        x = self._dist._process_quantiles(x, self.dim)
        U = self.shape_info.U
        log_pdet = self.shape_info.log_pdet
        return self._dist._logpdf(x, self.loc, U, log_pdet, self.df, self.dim, self.shape_info.rank)

    def cdf(self, x, *, maxpts=None, lower_limit=None, random_state=None):
        x = self._dist._process_quantiles(x, self.dim)
        return self._dist._cdf(x, self.loc, self.shape, self.df, self.dim, maxpts, lower_limit, random_state)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(loc=self.loc, shape=self.shape, df=self.df, size=size, random_state=random_state)

    def entropy(self):
        return self._dist._entropy(self.dim, self.df, self.shape)