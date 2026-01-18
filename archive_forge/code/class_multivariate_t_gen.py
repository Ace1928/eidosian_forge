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
class multivariate_t_gen(multi_rv_generic):
    """A multivariate t-distributed random variable.

    The `loc` parameter specifies the location. The `shape` parameter specifies
    the positive semidefinite shape matrix. The `df` parameter specifies the
    degrees of freedom.

    In addition to calling the methods below, the object itself may be called
    as a function to fix the location, shape matrix, and degrees of freedom
    parameters, returning a "frozen" multivariate t-distribution random.

    Methods
    -------
    pdf(x, loc=None, shape=1, df=1, allow_singular=False)
        Probability density function.
    logpdf(x, loc=None, shape=1, df=1, allow_singular=False)
        Log of the probability density function.
    cdf(x, loc=None, shape=1, df=1, allow_singular=False, *,
        maxpts=None, lower_limit=None, random_state=None)
        Cumulative distribution function.
    rvs(loc=None, shape=1, df=1, size=1, random_state=None)
        Draw random samples from a multivariate t-distribution.
    entropy(loc=None, shape=1, df=1)
        Differential entropy of a multivariate t-distribution.

    Parameters
    ----------
    %(_mvt_doc_default_callparams)s
    %(_doc_random_state)s

    Notes
    -----
    %(_mvt_doc_callparams_note)s
    The matrix `shape` must be a (symmetric) positive semidefinite matrix. The
    determinant and inverse of `shape` are computed as the pseudo-determinant
    and pseudo-inverse, respectively, so that `shape` does not need to have
    full rank.

    The probability density function for `multivariate_t` is

    .. math::

        f(x) = \\frac{\\Gamma((\\nu + p)/2)}{\\Gamma(\\nu/2)\\nu^{p/2}\\pi^{p/2}|\\Sigma|^{1/2}}
               \\left[1 + \\frac{1}{\\nu} (\\mathbf{x} - \\boldsymbol{\\mu})^{\\top}
               \\boldsymbol{\\Sigma}^{-1}
               (\\mathbf{x} - \\boldsymbol{\\mu}) \\right]^{-(\\nu + p)/2},

    where :math:`p` is the dimension of :math:`\\mathbf{x}`,
    :math:`\\boldsymbol{\\mu}` is the :math:`p`-dimensional location,
    :math:`\\boldsymbol{\\Sigma}` the :math:`p \\times p`-dimensional shape
    matrix, and :math:`\\nu` is the degrees of freedom.

    .. versionadded:: 1.6.0

    References
    ----------
    .. [1] Arellano-Valle et al. "Shannon Entropy and Mutual Information for
           Multivariate Skew-Elliptical Distributions". Scandinavian Journal
           of Statistics. Vol. 40, issue 1.

    Examples
    --------
    The object may be called (as a function) to fix the `loc`, `shape`,
    `df`, and `allow_singular` parameters, returning a "frozen"
    multivariate_t random variable:

    >>> import numpy as np
    >>> from scipy.stats import multivariate_t
    >>> rv = multivariate_t([1.0, -0.5], [[2.1, 0.3], [0.3, 1.5]], df=2)
    >>> # Frozen object with the same methods but holding the given location,
    >>> # scale, and degrees of freedom fixed.

    Create a contour plot of the PDF.

    >>> import matplotlib.pyplot as plt
    >>> x, y = np.mgrid[-1:3:.01, -2:1.5:.01]
    >>> pos = np.dstack((x, y))
    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.set_aspect('equal')
    >>> plt.contourf(x, y, rv.pdf(pos))

    """

    def __init__(self, seed=None):
        """Initialize a multivariate t-distributed random variable.

        Parameters
        ----------
        seed : Random state.

        """
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__, mvt_docdict_params)
        self._random_state = check_random_state(seed)

    def __call__(self, loc=None, shape=1, df=1, allow_singular=False, seed=None):
        """Create a frozen multivariate t-distribution.

        See `multivariate_t_frozen` for parameters.
        """
        if df == np.inf:
            return multivariate_normal_frozen(mean=loc, cov=shape, allow_singular=allow_singular, seed=seed)
        return multivariate_t_frozen(loc=loc, shape=shape, df=df, allow_singular=allow_singular, seed=seed)

    def pdf(self, x, loc=None, shape=1, df=1, allow_singular=False):
        """Multivariate t-distribution probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        %(_mvt_doc_default_callparams)s

        Returns
        -------
        pdf : Probability density function evaluated at `x`.

        Examples
        --------
        >>> from scipy.stats import multivariate_t
        >>> x = [0.4, 5]
        >>> loc = [0, 1]
        >>> shape = [[1, 0.1], [0.1, 1]]
        >>> df = 7
        >>> multivariate_t.pdf(x, loc, shape, df)
        0.00075713

        """
        dim, loc, shape, df = self._process_parameters(loc, shape, df)
        x = self._process_quantiles(x, dim)
        shape_info = _PSD(shape, allow_singular=allow_singular)
        logpdf = self._logpdf(x, loc, shape_info.U, shape_info.log_pdet, df, dim, shape_info.rank)
        return np.exp(logpdf)

    def logpdf(self, x, loc=None, shape=1, df=1):
        """Log of the multivariate t-distribution probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability density
            function.
        %(_mvt_doc_default_callparams)s

        Returns
        -------
        logpdf : Log of the probability density function evaluated at `x`.

        Examples
        --------
        >>> from scipy.stats import multivariate_t
        >>> x = [0.4, 5]
        >>> loc = [0, 1]
        >>> shape = [[1, 0.1], [0.1, 1]]
        >>> df = 7
        >>> multivariate_t.logpdf(x, loc, shape, df)
        -7.1859802

        See Also
        --------
        pdf : Probability density function.

        """
        dim, loc, shape, df = self._process_parameters(loc, shape, df)
        x = self._process_quantiles(x, dim)
        shape_info = _PSD(shape)
        return self._logpdf(x, loc, shape_info.U, shape_info.log_pdet, df, dim, shape_info.rank)

    def _logpdf(self, x, loc, prec_U, log_pdet, df, dim, rank):
        """Utility method `pdf`, `logpdf` for parameters.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability density
            function.
        loc : ndarray
            Location of the distribution.
        prec_U : ndarray
            A decomposition such that `np.dot(prec_U, prec_U.T)` is the inverse
            of the shape matrix.
        log_pdet : float
            Logarithm of the determinant of the shape matrix.
        df : float
            Degrees of freedom of the distribution.
        dim : int
            Dimension of the quantiles x.
        rank : int
            Rank of the shape matrix.

        Notes
        -----
        As this function does no argument checking, it should not be called
        directly; use 'logpdf' instead.

        """
        if df == np.inf:
            return multivariate_normal._logpdf(x, loc, prec_U, log_pdet, rank)
        dev = x - loc
        maha = np.square(np.dot(dev, prec_U)).sum(axis=-1)
        t = 0.5 * (df + dim)
        A = gammaln(t)
        B = gammaln(0.5 * df)
        C = dim / 2.0 * np.log(df * np.pi)
        D = 0.5 * log_pdet
        E = -t * np.log(1 + 1.0 / df * maha)
        return _squeeze_output(A - B - C - D + E)

    def _cdf(self, x, loc, shape, df, dim, maxpts=None, lower_limit=None, random_state=None):
        if random_state is not None:
            rng = check_random_state(random_state)
        else:
            rng = self._random_state
        if not maxpts:
            maxpts = 1000 * dim
        x = self._process_quantiles(x, dim)
        lower_limit = np.full(loc.shape, -np.inf) if lower_limit is None else lower_limit
        x, lower_limit = (x - loc, lower_limit - loc)
        b, a = np.broadcast_arrays(x, lower_limit)
        i_swap = b < a
        signs = (-1) ** i_swap.sum(axis=-1)
        a, b = (a.copy(), b.copy())
        a[i_swap], b[i_swap] = (b[i_swap], a[i_swap])
        n = x.shape[-1]
        limits = np.concatenate((a, b), axis=-1)

        def func1d(limits):
            a, b = (limits[:n], limits[n:])
            return _qmvt(maxpts, df, shape, a, b, rng)[0]
        res = np.apply_along_axis(func1d, -1, limits) * signs
        return _squeeze_output(res)

    def cdf(self, x, loc=None, shape=1, df=1, allow_singular=False, *, maxpts=None, lower_limit=None, random_state=None):
        """Multivariate t-distribution cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative distribution function.
        %(_mvt_doc_default_callparams)s
        maxpts : int, optional
            Maximum number of points to use for integration. The default is
            1000 times the number of dimensions.
        lower_limit : array_like, optional
            Lower limit of integration of the cumulative distribution function.
            Default is negative infinity. Must be broadcastable with `x`.
        %(_doc_random_state)s

        Returns
        -------
        cdf : ndarray or scalar
            Cumulative distribution function evaluated at `x`.

        Examples
        --------
        >>> from scipy.stats import multivariate_t
        >>> x = [0.4, 5]
        >>> loc = [0, 1]
        >>> shape = [[1, 0.1], [0.1, 1]]
        >>> df = 7
        >>> multivariate_t.cdf(x, loc, shape, df)
        0.64798491

        """
        dim, loc, shape, df = self._process_parameters(loc, shape, df)
        shape = _PSD(shape, allow_singular=allow_singular)._M
        return self._cdf(x, loc, shape, df, dim, maxpts, lower_limit, random_state)

    def _entropy(self, dim, df=1, shape=1):
        if df == np.inf:
            return multivariate_normal(None, cov=shape).entropy()
        shape_info = _PSD(shape)
        shape_term = 0.5 * shape_info.log_pdet

        def regular(dim, df):
            halfsum = 0.5 * (dim + df)
            half_df = 0.5 * df
            return -gammaln(halfsum) + gammaln(half_df) + 0.5 * dim * np.log(df * np.pi) + halfsum * (psi(halfsum) - psi(half_df)) + shape_term

        def asymptotic(dim, df):
            return (dim * norm._entropy() + dim / df - dim * (dim - 2) * df ** (-2.0) / 4 + dim ** 2 * (dim - 2) * df ** (-3.0) / 6 + dim * (-3 * dim ** 3 + 8 * dim ** 2 - 8) * df ** (-4.0) / 24 + dim ** 2 * (3 * dim ** 3 - 10 * dim ** 2 + 16) * df ** (-5.0) / 30 + shape_term)[()]
        threshold = dim * 100 * 4 / (np.log(dim) + 1)
        return _lazywhere(df >= threshold, (dim, df), f=asymptotic, f2=regular)

    def entropy(self, loc=None, shape=1, df=1):
        """Calculate the differential entropy of a multivariate
        t-distribution.

        Parameters
        ----------
        %(_mvt_doc_default_callparams)s

        Returns
        -------
        h : float
            Differential entropy

        """
        dim, loc, shape, df = self._process_parameters(None, shape, df)
        return self._entropy(dim, df, shape)

    def rvs(self, loc=None, shape=1, df=1, size=1, random_state=None):
        """Draw random samples from a multivariate t-distribution.

        Parameters
        ----------
        %(_mvt_doc_default_callparams)s
        size : integer, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `P`), where `P` is the
            dimension of the random variable.

        Examples
        --------
        >>> from scipy.stats import multivariate_t
        >>> x = [0.4, 5]
        >>> loc = [0, 1]
        >>> shape = [[1, 0.1], [0.1, 1]]
        >>> df = 7
        >>> multivariate_t.rvs(loc, shape, df)
        array([[0.93477495, 3.00408716]])

        """
        dim, loc, shape, df = self._process_parameters(loc, shape, df)
        if random_state is not None:
            rng = check_random_state(random_state)
        else:
            rng = self._random_state
        if np.isinf(df):
            x = np.ones(size)
        else:
            x = rng.chisquare(df, size=size) / df
        z = rng.multivariate_normal(np.zeros(dim), shape, size=size)
        samples = loc + z / np.sqrt(x)[..., None]
        return _squeeze_output(samples)

    def _process_quantiles(self, x, dim):
        """
        Adjust quantiles array so that last axis labels the components of
        each data point.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = x[np.newaxis]
        elif x.ndim == 1:
            if dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]
        return x

    def _process_parameters(self, loc, shape, df):
        """
        Infer dimensionality from location array and shape matrix, handle
        defaults, and ensure compatible dimensions.
        """
        if loc is None and shape is None:
            loc = np.asarray(0, dtype=float)
            shape = np.asarray(1, dtype=float)
            dim = 1
        elif loc is None:
            shape = np.asarray(shape, dtype=float)
            if shape.ndim < 2:
                dim = 1
            else:
                dim = shape.shape[0]
            loc = np.zeros(dim)
        elif shape is None:
            loc = np.asarray(loc, dtype=float)
            dim = loc.size
            shape = np.eye(dim)
        else:
            shape = np.asarray(shape, dtype=float)
            loc = np.asarray(loc, dtype=float)
            dim = loc.size
        if dim == 1:
            loc = loc.reshape(1)
            shape = shape.reshape(1, 1)
        if loc.ndim != 1 or loc.shape[0] != dim:
            raise ValueError("Array 'loc' must be a vector of length %d." % dim)
        if shape.ndim == 0:
            shape = shape * np.eye(dim)
        elif shape.ndim == 1:
            shape = np.diag(shape)
        elif shape.ndim == 2 and shape.shape != (dim, dim):
            rows, cols = shape.shape
            if rows != cols:
                msg = "Array 'cov' must be square if it is two dimensional, but cov.shape = %s." % str(shape.shape)
            else:
                msg = "Dimension mismatch: array 'cov' is of shape %s, but 'loc' is a vector of length %d."
                msg = msg % (str(shape.shape), len(loc))
            raise ValueError(msg)
        elif shape.ndim > 2:
            raise ValueError("Array 'cov' must be at most two-dimensional, but cov.ndim = %d" % shape.ndim)
        if df is None:
            df = 1
        elif df <= 0:
            raise ValueError("'df' must be greater than zero.")
        elif np.isnan(df):
            raise ValueError("'df' is 'nan' but must be greater than zero or 'np.inf'.")
        return (dim, loc, shape, df)