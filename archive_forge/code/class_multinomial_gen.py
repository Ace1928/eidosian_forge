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
class multinomial_gen(multi_rv_generic):
    """A multinomial random variable.

    Methods
    -------
    pmf(x, n, p)
        Probability mass function.
    logpmf(x, n, p)
        Log of the probability mass function.
    rvs(n, p, size=1, random_state=None)
        Draw random samples from a multinomial distribution.
    entropy(n, p)
        Compute the entropy of the multinomial distribution.
    cov(n, p)
        Compute the covariance matrix of the multinomial distribution.

    Parameters
    ----------
    %(_doc_default_callparams)s
    %(_doc_random_state)s

    Notes
    -----
    %(_doc_callparams_note)s

    The probability mass function for `multinomial` is

    .. math::

        f(x) = \\frac{n!}{x_1! \\cdots x_k!} p_1^{x_1} \\cdots p_k^{x_k},

    supported on :math:`x=(x_1, \\ldots, x_k)` where each :math:`x_i` is a
    nonnegative integer and their sum is :math:`n`.

    .. versionadded:: 0.19.0

    Examples
    --------

    >>> from scipy.stats import multinomial
    >>> rv = multinomial(8, [0.3, 0.2, 0.5])
    >>> rv.pmf([1, 3, 4])
    0.042000000000000072

    The multinomial distribution for :math:`k=2` is identical to the
    corresponding binomial distribution (tiny numerical differences
    notwithstanding):

    >>> from scipy.stats import binom
    >>> multinomial.pmf([3, 4], n=7, p=[0.4, 0.6])
    0.29030399999999973
    >>> binom.pmf(3, 7, 0.4)
    0.29030400000000012

    The functions ``pmf``, ``logpmf``, ``entropy``, and ``cov`` support
    broadcasting, under the convention that the vector parameters (``x`` and
    ``p``) are interpreted as if each row along the last axis is a single
    object. For instance:

    >>> multinomial.pmf([[3, 4], [3, 5]], n=[7, 8], p=[.3, .7])
    array([0.2268945,  0.25412184])

    Here, ``x.shape == (2, 2)``, ``n.shape == (2,)``, and ``p.shape == (2,)``,
    but following the rules mentioned above they behave as if the rows
    ``[3, 4]`` and ``[3, 5]`` in ``x`` and ``[.3, .7]`` in ``p`` were a single
    object, and as if we had ``x.shape = (2,)``, ``n.shape = (2,)``, and
    ``p.shape = ()``. To obtain the individual elements without broadcasting,
    we would do this:

    >>> multinomial.pmf([3, 4], n=7, p=[.3, .7])
    0.2268945
    >>> multinomial.pmf([3, 5], 8, p=[.3, .7])
    0.25412184

    This broadcasting also works for ``cov``, where the output objects are
    square matrices of size ``p.shape[-1]``. For example:

    >>> multinomial.cov([4, 5], [[.3, .7], [.4, .6]])
    array([[[ 0.84, -0.84],
            [-0.84,  0.84]],
           [[ 1.2 , -1.2 ],
            [-1.2 ,  1.2 ]]])

    In this example, ``n.shape == (2,)`` and ``p.shape == (2, 2)``, and
    following the rules above, these broadcast as if ``p.shape == (2,)``.
    Thus the result should also be of shape ``(2,)``, but since each output is
    a :math:`2 \\times 2` matrix, the result in fact has shape ``(2, 2, 2)``,
    where ``result[0]`` is equal to ``multinomial.cov(n=4, p=[.3, .7])`` and
    ``result[1]`` is equal to ``multinomial.cov(n=5, p=[.4, .6])``.

    Alternatively, the object may be called (as a function) to fix the `n` and
    `p` parameters, returning a "frozen" multinomial random variable:

    >>> rv = multinomial(n=7, p=[.3, .7])
    >>> # Frozen object with the same methods but holding the given
    >>> # degrees of freedom and scale fixed.

    See also
    --------
    scipy.stats.binom : The binomial distribution.
    numpy.random.Generator.multinomial : Sampling from the multinomial distribution.
    scipy.stats.multivariate_hypergeom :
        The multivariate hypergeometric distribution.
    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__, multinomial_docdict_params)

    def __call__(self, n, p, seed=None):
        """Create a frozen multinomial distribution.

        See `multinomial_frozen` for more information.
        """
        return multinomial_frozen(n, p, seed)

    def _process_parameters(self, n, p, eps=1e-15):
        """Returns: n_, p_, npcond.

        n_ and p_ are arrays of the correct shape; npcond is a boolean array
        flagging values out of the domain.
        """
        p = np.array(p, dtype=np.float64, copy=True)
        p_adjusted = 1.0 - p[..., :-1].sum(axis=-1)
        i_adjusted = np.abs(p_adjusted) > eps
        p[i_adjusted, -1] = p_adjusted[i_adjusted]
        pcond = np.any(p < 0, axis=-1)
        pcond |= np.any(p > 1, axis=-1)
        n = np.array(n, dtype=int, copy=True)
        ncond = n < 0
        return (n, p, ncond | pcond)

    def _process_quantiles(self, x, n, p):
        """Returns: x_, xcond.

        x_ is an int array; xcond is a boolean array flagging values out of the
        domain.
        """
        xx = np.asarray(x, dtype=int)
        if xx.ndim == 0:
            raise ValueError('x must be an array.')
        if xx.size != 0 and (not xx.shape[-1] == p.shape[-1]):
            raise ValueError('Size of each quantile should be size of p: received %d, but expected %d.' % (xx.shape[-1], p.shape[-1]))
        cond = np.any(xx != x, axis=-1)
        cond |= np.any(xx < 0, axis=-1)
        cond = cond | (np.sum(xx, axis=-1) != n)
        return (xx, cond)

    def _checkresult(self, result, cond, bad_value):
        result = np.asarray(result)
        if cond.ndim != 0:
            result[cond] = bad_value
        elif cond:
            if result.ndim == 0:
                return bad_value
            result[...] = bad_value
        return result

    def _logpmf(self, x, n, p):
        return gammaln(n + 1) + np.sum(xlogy(x, p) - gammaln(x + 1), axis=-1)

    def logpmf(self, x, n, p):
        """Log of the Multinomial probability mass function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_doc_default_callparams)s

        Returns
        -------
        logpmf : ndarray or scalar
            Log of the probability mass function evaluated at `x`

        Notes
        -----
        %(_doc_callparams_note)s
        """
        n, p, npcond = self._process_parameters(n, p)
        x, xcond = self._process_quantiles(x, n, p)
        result = self._logpmf(x, n, p)
        xcond_ = xcond | np.zeros(npcond.shape, dtype=np.bool_)
        result = self._checkresult(result, xcond_, -np.inf)
        npcond_ = npcond | np.zeros(xcond.shape, dtype=np.bool_)
        return self._checkresult(result, npcond_, np.nan)

    def pmf(self, x, n, p):
        """Multinomial probability mass function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_doc_default_callparams)s

        Returns
        -------
        pmf : ndarray or scalar
            Probability density function evaluated at `x`

        Notes
        -----
        %(_doc_callparams_note)s
        """
        return np.exp(self.logpmf(x, n, p))

    def mean(self, n, p):
        """Mean of the Multinomial distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        mean : float
            The mean of the distribution
        """
        n, p, npcond = self._process_parameters(n, p)
        result = n[..., np.newaxis] * p
        return self._checkresult(result, npcond, np.nan)

    def cov(self, n, p):
        """Covariance matrix of the multinomial distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        cov : ndarray
            The covariance matrix of the distribution
        """
        n, p, npcond = self._process_parameters(n, p)
        nn = n[..., np.newaxis, np.newaxis]
        result = nn * np.einsum('...j,...k->...jk', -p, p)
        for i in range(p.shape[-1]):
            result[..., i, i] += n * p[..., i]
        return self._checkresult(result, npcond, np.nan)

    def entropy(self, n, p):
        """Compute the entropy of the multinomial distribution.

        The entropy is computed using this expression:

        .. math::

            f(x) = - \\log n! - n\\sum_{i=1}^k p_i \\log p_i +
            \\sum_{i=1}^k \\sum_{x=0}^n \\binom n x p_i^x(1-p_i)^{n-x} \\log x!

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        h : scalar
            Entropy of the Multinomial distribution

        Notes
        -----
        %(_doc_callparams_note)s
        """
        n, p, npcond = self._process_parameters(n, p)
        x = np.r_[1:np.max(n) + 1]
        term1 = n * np.sum(entr(p), axis=-1)
        term1 -= gammaln(n + 1)
        n = n[..., np.newaxis]
        new_axes_needed = max(p.ndim, n.ndim) - x.ndim + 1
        x.shape += (1,) * new_axes_needed
        term2 = np.sum(binom.pmf(x, n, p) * gammaln(x + 1), axis=(-1, -1 - new_axes_needed))
        return self._checkresult(term1 + term2, npcond, np.nan)

    def rvs(self, n, p, size=None, random_state=None):
        """Draw random samples from a Multinomial distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s
        size : integer or iterable of integers, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of shape (`size`, `len(p)`)

        Notes
        -----
        %(_doc_callparams_note)s
        """
        n, p, npcond = self._process_parameters(n, p)
        random_state = self._get_random_state(random_state)
        return random_state.multinomial(n, p, size)