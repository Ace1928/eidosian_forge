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
class dirichlet_gen(multi_rv_generic):
    """A Dirichlet random variable.

    The ``alpha`` keyword specifies the concentration parameters of the
    distribution.

    .. versionadded:: 0.15.0

    Methods
    -------
    pdf(x, alpha)
        Probability density function.
    logpdf(x, alpha)
        Log of the probability density function.
    rvs(alpha, size=1, random_state=None)
        Draw random samples from a Dirichlet distribution.
    mean(alpha)
        The mean of the Dirichlet distribution
    var(alpha)
        The variance of the Dirichlet distribution
    cov(alpha)
        The covariance of the Dirichlet distribution
    entropy(alpha)
        Compute the differential entropy of the Dirichlet distribution.

    Parameters
    ----------
    %(_dirichlet_doc_default_callparams)s
    %(_doc_random_state)s

    Notes
    -----
    Each :math:`\\alpha` entry must be positive. The distribution has only
    support on the simplex defined by

    .. math::
        \\sum_{i=1}^{K} x_i = 1

    where :math:`0 < x_i < 1`.

    If the quantiles don't lie within the simplex, a ValueError is raised.

    The probability density function for `dirichlet` is

    .. math::

        f(x) = \\frac{1}{\\mathrm{B}(\\boldsymbol\\alpha)} \\prod_{i=1}^K x_i^{\\alpha_i - 1}

    where

    .. math::

        \\mathrm{B}(\\boldsymbol\\alpha) = \\frac{\\prod_{i=1}^K \\Gamma(\\alpha_i)}
                                     {\\Gamma\\bigl(\\sum_{i=1}^K \\alpha_i\\bigr)}

    and :math:`\\boldsymbol\\alpha=(\\alpha_1,\\ldots,\\alpha_K)`, the
    concentration parameters and :math:`K` is the dimension of the space
    where :math:`x` takes values.

    Note that the `dirichlet` interface is somewhat inconsistent.
    The array returned by the rvs function is transposed
    with respect to the format expected by the pdf and logpdf.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import dirichlet

    Generate a dirichlet random variable

    >>> quantiles = np.array([0.2, 0.2, 0.6])  # specify quantiles
    >>> alpha = np.array([0.4, 5, 15])  # specify concentration parameters
    >>> dirichlet.pdf(quantiles, alpha)
    0.2843831684937255

    The same PDF but following a log scale

    >>> dirichlet.logpdf(quantiles, alpha)
    -1.2574327653159187

    Once we specify the dirichlet distribution
    we can then calculate quantities of interest

    >>> dirichlet.mean(alpha)  # get the mean of the distribution
    array([0.01960784, 0.24509804, 0.73529412])
    >>> dirichlet.var(alpha) # get variance
    array([0.00089829, 0.00864603, 0.00909517])
    >>> dirichlet.entropy(alpha)  # calculate the differential entropy
    -4.3280162474082715

    We can also return random samples from the distribution

    >>> dirichlet.rvs(alpha, size=1, random_state=1)
    array([[0.00766178, 0.24670518, 0.74563305]])
    >>> dirichlet.rvs(alpha, size=2, random_state=2)
    array([[0.01639427, 0.1292273 , 0.85437844],
           [0.00156917, 0.19033695, 0.80809388]])

    Alternatively, the object may be called (as a function) to fix
    concentration parameters, returning a "frozen" Dirichlet
    random variable:

    >>> rv = dirichlet(alpha)
    >>> # Frozen object with the same methods but holding the given
    >>> # concentration parameters fixed.

    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__, dirichlet_docdict_params)

    def __call__(self, alpha, seed=None):
        return dirichlet_frozen(alpha, seed=seed)

    def _logpdf(self, x, alpha):
        """Log of the Dirichlet probability density function.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function
        %(_dirichlet_doc_default_callparams)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.

        """
        lnB = _lnB(alpha)
        return -lnB + np.sum(xlogy(alpha - 1, x.T).T, 0)

    def logpdf(self, x, alpha):
        """Log of the Dirichlet probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_dirichlet_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray or scalar
            Log of the probability density function evaluated at `x`.

        """
        alpha = _dirichlet_check_parameters(alpha)
        x = _dirichlet_check_input(alpha, x)
        out = self._logpdf(x, alpha)
        return _squeeze_output(out)

    def pdf(self, x, alpha):
        """The Dirichlet probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_dirichlet_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray or scalar
            The probability density function evaluated at `x`.

        """
        alpha = _dirichlet_check_parameters(alpha)
        x = _dirichlet_check_input(alpha, x)
        out = np.exp(self._logpdf(x, alpha))
        return _squeeze_output(out)

    def mean(self, alpha):
        """Mean of the Dirichlet distribution.

        Parameters
        ----------
        %(_dirichlet_doc_default_callparams)s

        Returns
        -------
        mu : ndarray or scalar
            Mean of the Dirichlet distribution.

        """
        alpha = _dirichlet_check_parameters(alpha)
        out = alpha / np.sum(alpha)
        return _squeeze_output(out)

    def var(self, alpha):
        """Variance of the Dirichlet distribution.

        Parameters
        ----------
        %(_dirichlet_doc_default_callparams)s

        Returns
        -------
        v : ndarray or scalar
            Variance of the Dirichlet distribution.

        """
        alpha = _dirichlet_check_parameters(alpha)
        alpha0 = np.sum(alpha)
        out = alpha * (alpha0 - alpha) / (alpha0 * alpha0 * (alpha0 + 1))
        return _squeeze_output(out)

    def cov(self, alpha):
        """Covariance matrix of the Dirichlet distribution.

        Parameters
        ----------
        %(_dirichlet_doc_default_callparams)s

        Returns
        -------
        cov : ndarray
            The covariance matrix of the distribution.
        """
        alpha = _dirichlet_check_parameters(alpha)
        alpha0 = np.sum(alpha)
        a = alpha / alpha0
        cov = (np.diag(a) - np.outer(a, a)) / (alpha0 + 1)
        return _squeeze_output(cov)

    def entropy(self, alpha):
        """
        Differential entropy of the Dirichlet distribution.

        Parameters
        ----------
        %(_dirichlet_doc_default_callparams)s

        Returns
        -------
        h : scalar
            Entropy of the Dirichlet distribution

        """
        alpha = _dirichlet_check_parameters(alpha)
        alpha0 = np.sum(alpha)
        lnB = _lnB(alpha)
        K = alpha.shape[0]
        out = lnB + (alpha0 - K) * scipy.special.psi(alpha0) - np.sum((alpha - 1) * scipy.special.psi(alpha))
        return _squeeze_output(out)

    def rvs(self, alpha, size=1, random_state=None):
        """
        Draw random samples from a Dirichlet distribution.

        Parameters
        ----------
        %(_dirichlet_doc_default_callparams)s
        size : int, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.

        """
        alpha = _dirichlet_check_parameters(alpha)
        random_state = self._get_random_state(random_state)
        return random_state.dirichlet(alpha, size=size)