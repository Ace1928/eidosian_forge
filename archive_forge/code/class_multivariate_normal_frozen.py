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
class multivariate_normal_frozen(multi_rv_frozen):

    def __init__(self, mean=None, cov=1, allow_singular=False, seed=None, maxpts=None, abseps=1e-05, releps=1e-05):
        """Create a frozen multivariate normal distribution.

        Parameters
        ----------
        mean : array_like, default: ``[0]``
            Mean of the distribution.
        cov : array_like, default: ``[1]``
            Symmetric positive (semi)definite covariance matrix of the
            distribution.
        allow_singular : bool, default: ``False``
            Whether to allow a singular covariance matrix.
        seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.
        maxpts : integer, optional
            The maximum number of points to use for integration of the
            cumulative distribution function (default `1000000*dim`)
        abseps : float, optional
            Absolute error tolerance for the cumulative distribution function
            (default 1e-5)
        releps : float, optional
            Relative error tolerance for the cumulative distribution function
            (default 1e-5)

        Examples
        --------
        When called with the default parameters, this will create a 1D random
        variable with mean 0 and covariance 1:

        >>> from scipy.stats import multivariate_normal
        >>> r = multivariate_normal()
        >>> r.mean
        array([ 0.])
        >>> r.cov
        array([[1.]])

        """
        self._dist = multivariate_normal_gen(seed)
        self.dim, self.mean, self.cov_object = self._dist._process_parameters(mean, cov, allow_singular)
        self.allow_singular = allow_singular or self.cov_object._allow_singular
        if not maxpts:
            maxpts = 1000000 * self.dim
        self.maxpts = maxpts
        self.abseps = abseps
        self.releps = releps

    @property
    def cov(self):
        return self.cov_object.covariance

    def logpdf(self, x):
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._logpdf(x, self.mean, self.cov_object)
        if np.any(self.cov_object.rank < self.dim):
            out_of_bounds = ~self.cov_object._support_mask(x - self.mean)
            out[out_of_bounds] = -np.inf
        return _squeeze_output(out)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def logcdf(self, x, *, lower_limit=None):
        cdf = self.cdf(x, lower_limit=lower_limit)
        cdf = cdf + 0j if np.any(cdf < 0) else cdf
        out = np.log(cdf)
        return out

    def cdf(self, x, *, lower_limit=None):
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._cdf(x, self.mean, self.cov_object.covariance, self.maxpts, self.abseps, self.releps, lower_limit)
        return _squeeze_output(out)

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.mean, self.cov_object, size, random_state)

    def entropy(self):
        """Computes the differential entropy of the multivariate normal.

        Returns
        -------
        h : scalar
            Entropy of the multivariate normal distribution

        """
        log_pdet = self.cov_object.log_pdet
        rank = self.cov_object.rank
        return 0.5 * (rank * (_LOG_2PI + 1) + log_pdet)