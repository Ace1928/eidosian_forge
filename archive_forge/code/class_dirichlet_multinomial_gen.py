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
class dirichlet_multinomial_gen(multi_rv_generic):
    """A Dirichlet multinomial random variable.

    The Dirichlet multinomial distribution is a compound probability
    distribution: it is the multinomial distribution with number of trials
    `n` and class probabilities ``p`` randomly sampled from a Dirichlet
    distribution with concentration parameters ``alpha``.

    Methods
    -------
    logpmf(x, alpha, n):
        Log of the probability mass function.
    pmf(x, alpha, n):
        Probability mass function.
    mean(alpha, n):
        Mean of the Dirichlet multinomial distribution.
    var(alpha, n):
        Variance of the Dirichlet multinomial distribution.
    cov(alpha, n):
        The covariance of the Dirichlet multinomial distribution.

    Parameters
    ----------
    %(_dirichlet_mn_doc_default_callparams)s
    %(_doc_random_state)s

    See Also
    --------
    scipy.stats.dirichlet : The dirichlet distribution.
    scipy.stats.multinomial : The multinomial distribution.

    References
    ----------
    .. [1] Dirichlet-multinomial distribution, Wikipedia,
           https://www.wikipedia.org/wiki/Dirichlet-multinomial_distribution

    Examples
    --------
    >>> from scipy.stats import dirichlet_multinomial

    Get the PMF

    >>> n = 6  # number of trials
    >>> alpha = [3, 4, 5]  # concentration parameters
    >>> x = [1, 2, 3]  # counts
    >>> dirichlet_multinomial.pmf(x, alpha, n)
    0.08484162895927604

    If the sum of category counts does not equal the number of trials,
    the probability mass is zero.

    >>> dirichlet_multinomial.pmf(x, alpha, n=7)
    0.0

    Get the log of the PMF

    >>> dirichlet_multinomial.logpmf(x, alpha, n)
    -2.4669689491013327

    Get the mean

    >>> dirichlet_multinomial.mean(alpha, n)
    array([1.5, 2. , 2.5])

    Get the variance

    >>> dirichlet_multinomial.var(alpha, n)
    array([1.55769231, 1.84615385, 2.01923077])

    Get the covariance

    >>> dirichlet_multinomial.cov(alpha, n)
    array([[ 1.55769231, -0.69230769, -0.86538462],
           [-0.69230769,  1.84615385, -1.15384615],
           [-0.86538462, -1.15384615,  2.01923077]])

    Alternatively, the object may be called (as a function) to fix the
    `alpha` and `n` parameters, returning a "frozen" Dirichlet multinomial
    random variable.

    >>> dm = dirichlet_multinomial(alpha, n)
    >>> dm.pmf(x)
    0.08484162895927579

    All methods are fully vectorized. Each element of `x` and `alpha` is
    a vector (along the last axis), each element of `n` is an
    integer (scalar), and the result is computed element-wise.

    >>> x = [[1, 2, 3], [4, 5, 6]]
    >>> alpha = [[1, 2, 3], [4, 5, 6]]
    >>> n = [6, 15]
    >>> dirichlet_multinomial.pmf(x, alpha, n)
    array([0.06493506, 0.02626937])

    >>> dirichlet_multinomial.cov(alpha, n).shape  # both covariance matrices
    (2, 3, 3)

    Broadcasting according to standard NumPy conventions is supported. Here,
    we have four sets of concentration parameters (each a two element vector)
    for each of three numbers of trials (each a scalar).

    >>> alpha = [[3, 4], [4, 5], [5, 6], [6, 7]]
    >>> n = [[6], [7], [8]]
    >>> dirichlet_multinomial.mean(alpha, n).shape
    (3, 4, 2)

    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__, dirichlet_mn_docdict_params)

    def __call__(self, alpha, n, seed=None):
        return dirichlet_multinomial_frozen(alpha, n, seed=seed)

    def logpmf(self, x, alpha, n):
        """The log of the probability mass function.

        Parameters
        ----------
        x: ndarray
            Category counts (non-negative integers). Must be broadcastable
            with shape parameter ``alpha``. If multidimensional, the last axis
            must correspond with the categories.
        %(_dirichlet_mn_doc_default_callparams)s

        Returns
        -------
        out: ndarray or scalar
            Log of the probability mass function.

        """
        a, Sa, n, x = _dirichlet_multinomial_check_parameters(alpha, n, x)
        out = np.asarray(loggamma(Sa) + loggamma(n + 1) - loggamma(n + Sa))
        out += (loggamma(x + a) - (loggamma(a) + loggamma(x + 1))).sum(axis=-1)
        np.place(out, n != x.sum(axis=-1), -np.inf)
        return out[()]

    def pmf(self, x, alpha, n):
        """Probability mass function for a Dirichlet multinomial distribution.

        Parameters
        ----------
        x: ndarray
            Category counts (non-negative integers). Must be broadcastable
            with shape parameter ``alpha``. If multidimensional, the last axis
            must correspond with the categories.
        %(_dirichlet_mn_doc_default_callparams)s

        Returns
        -------
        out: ndarray or scalar
            Probability mass function.

        """
        return np.exp(self.logpmf(x, alpha, n))

    def mean(self, alpha, n):
        """Mean of a Dirichlet multinomial distribution.

        Parameters
        ----------
        %(_dirichlet_mn_doc_default_callparams)s

        Returns
        -------
        out: ndarray
            Mean of a Dirichlet multinomial distribution.

        """
        a, Sa, n = _dirichlet_multinomial_check_parameters(alpha, n)
        n, Sa = (n[..., np.newaxis], Sa[..., np.newaxis])
        return n * a / Sa

    def var(self, alpha, n):
        """The variance of the Dirichlet multinomial distribution.

        Parameters
        ----------
        %(_dirichlet_mn_doc_default_callparams)s

        Returns
        -------
        out: array_like
            The variances of the components of the distribution. This is
            the diagonal of the covariance matrix of the distribution.

        """
        a, Sa, n = _dirichlet_multinomial_check_parameters(alpha, n)
        n, Sa = (n[..., np.newaxis], Sa[..., np.newaxis])
        return n * a / Sa * (1 - a / Sa) * (n + Sa) / (1 + Sa)

    def cov(self, alpha, n):
        """Covariance matrix of a Dirichlet multinomial distribution.

        Parameters
        ----------
        %(_dirichlet_mn_doc_default_callparams)s

        Returns
        -------
        out : array_like
            The covariance matrix of the distribution.

        """
        a, Sa, n = _dirichlet_multinomial_check_parameters(alpha, n)
        var = dirichlet_multinomial.var(a, n)
        n, Sa = (n[..., np.newaxis, np.newaxis], Sa[..., np.newaxis, np.newaxis])
        aiaj = a[..., :, np.newaxis] * a[..., np.newaxis, :]
        cov = -n * aiaj / Sa ** 2 * (n + Sa) / (1 + Sa)
        ii = np.arange(cov.shape[-1])
        cov[..., ii, ii] = var
        return cov