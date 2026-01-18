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
class vonmises_fisher_frozen(multi_rv_frozen):

    def __init__(self, mu=None, kappa=1, seed=None):
        """Create a frozen von Mises-Fisher distribution.

        Parameters
        ----------
        mu : array_like, default: None
            Mean direction of the distribution.
        kappa : float, default: 1
            Concentration parameter. Must be positive.
        seed : {None, int, `numpy.random.Generator`,
                `numpy.random.RandomState`}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        """
        self._dist = vonmises_fisher_gen(seed)
        self.dim, self.mu, self.kappa = self._dist._process_parameters(mu, kappa)

    def logpdf(self, x):
        """
        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability
            density function. The last axis of `x` must correspond
            to unit vectors of the same dimensionality as the distribution.

        Returns
        -------
        logpdf : ndarray or scalar
            Log of probability density function evaluated at `x`.

        """
        return self._dist._logpdf(x, self.dim, self.mu, self.kappa)

    def pdf(self, x):
        """
        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability
            density function. The last axis of `x` must correspond
            to unit vectors of the same dimensionality as the distribution.

        Returns
        -------
        pdf : ndarray or scalar
            Probability density function evaluated at `x`.

        """
        return np.exp(self.logpdf(x))

    def rvs(self, size=1, random_state=None):
        """Draw random variates from the Von Mises-Fisher distribution.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Given a shape of, for example, (m,n,k), m*n*k samples are
            generated, and packed in an m-by-n-by-k arrangement.
            Because each sample is N-dimensional, the output shape
            is (m,n,k,N). If no shape is specified, a single (N-D)
            sample is returned.
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
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the distribution.

        """
        random_state = self._dist._get_random_state(random_state)
        return self._dist._rvs(self.dim, self.mu, self.kappa, size, random_state)

    def entropy(self):
        """
        Calculate the differential entropy of the von Mises-Fisher
        distribution.

        Returns
        -------
        h: float
            Entropy of the Von Mises-Fisher distribution.

        """
        return self._dist._entropy(self.dim, self.kappa)