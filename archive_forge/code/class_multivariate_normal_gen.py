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
class multivariate_normal_gen(multi_rv_generic):
    """A multivariate normal random variable.

    The `mean` keyword specifies the mean. The `cov` keyword specifies the
    covariance matrix.

    Methods
    -------
    pdf(x, mean=None, cov=1, allow_singular=False)
        Probability density function.
    logpdf(x, mean=None, cov=1, allow_singular=False)
        Log of the probability density function.
    cdf(x, mean=None, cov=1, allow_singular=False, maxpts=1000000*dim, abseps=1e-5, releps=1e-5, lower_limit=None)
        Cumulative distribution function.
    logcdf(x, mean=None, cov=1, allow_singular=False, maxpts=1000000*dim, abseps=1e-5, releps=1e-5)
        Log of the cumulative distribution function.
    rvs(mean=None, cov=1, size=1, random_state=None)
        Draw random samples from a multivariate normal distribution.
    entropy(mean=None, cov=1)
        Compute the differential entropy of the multivariate normal.
    fit(x, fix_mean=None, fix_cov=None)
        Fit a multivariate normal distribution to data.

    Parameters
    ----------
    %(_mvn_doc_default_callparams)s
    %(_doc_random_state)s

    Notes
    -----
    %(_mvn_doc_callparams_note)s

    The covariance matrix `cov` may be an instance of a subclass of
    `Covariance`, e.g. `scipy.stats.CovViaPrecision`. If so, `allow_singular`
    is ignored.

    Otherwise, `cov` must be a symmetric positive semidefinite
    matrix when `allow_singular` is True; it must be (strictly) positive
    definite when `allow_singular` is False.
    Symmetry is not checked; only the lower triangular portion is used.
    The determinant and inverse of `cov` are computed
    as the pseudo-determinant and pseudo-inverse, respectively, so
    that `cov` does not need to have full rank.

    The probability density function for `multivariate_normal` is

    .. math::

        f(x) = \\frac{1}{\\sqrt{(2 \\pi)^k \\det \\Sigma}}
               \\exp\\left( -\\frac{1}{2} (x - \\mu)^T \\Sigma^{-1} (x - \\mu) \\right),

    where :math:`\\mu` is the mean, :math:`\\Sigma` the covariance matrix,
    :math:`k` the rank of :math:`\\Sigma`. In case of singular :math:`\\Sigma`,
    SciPy extends this definition according to [1]_.

    .. versionadded:: 0.14.0

    References
    ----------
    .. [1] Multivariate Normal Distribution - Degenerate Case, Wikipedia,
           https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Degenerate_case

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import multivariate_normal

    >>> x = np.linspace(0, 5, 10, endpoint=False)
    >>> y = multivariate_normal.pdf(x, mean=2.5, cov=0.5); y
    array([ 0.00108914,  0.01033349,  0.05946514,  0.20755375,  0.43939129,
            0.56418958,  0.43939129,  0.20755375,  0.05946514,  0.01033349])
    >>> fig1 = plt.figure()
    >>> ax = fig1.add_subplot(111)
    >>> ax.plot(x, y)
    >>> plt.show()

    Alternatively, the object may be called (as a function) to fix the mean
    and covariance parameters, returning a "frozen" multivariate normal
    random variable:

    >>> rv = multivariate_normal(mean=None, cov=1, allow_singular=False)
    >>> # Frozen object with the same methods but holding the given
    >>> # mean and covariance fixed.

    The input quantiles can be any shape of array, as long as the last
    axis labels the components.  This allows us for instance to
    display the frozen pdf for a non-isotropic random variable in 2D as
    follows:

    >>> x, y = np.mgrid[-1:1:.01, -1:1:.01]
    >>> pos = np.dstack((x, y))
    >>> rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    >>> fig2 = plt.figure()
    >>> ax2 = fig2.add_subplot(111)
    >>> ax2.contourf(x, y, rv.pdf(pos))

    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__, mvn_docdict_params)

    def __call__(self, mean=None, cov=1, allow_singular=False, seed=None):
        """Create a frozen multivariate normal distribution.

        See `multivariate_normal_frozen` for more information.
        """
        return multivariate_normal_frozen(mean, cov, allow_singular=allow_singular, seed=seed)

    def _process_parameters(self, mean, cov, allow_singular=True):
        """
        Infer dimensionality from mean or covariance matrix, ensure that
        mean and covariance are full vector resp. matrix.
        """
        if isinstance(cov, _covariance.Covariance):
            return self._process_parameters_Covariance(mean, cov)
        else:
            dim, mean, cov = self._process_parameters_psd(None, mean, cov)
            psd = _PSD(cov, allow_singular=allow_singular)
            cov_object = _covariance.CovViaPSD(psd)
            return (dim, mean, cov_object)

    def _process_parameters_Covariance(self, mean, cov):
        dim = cov.shape[-1]
        mean = np.array([0.0]) if mean is None else mean
        message = f'`cov` represents a covariance matrix in {dim} dimensions,and so `mean` must be broadcastable to shape {(dim,)}'
        try:
            mean = np.broadcast_to(mean, dim)
        except ValueError as e:
            raise ValueError(message) from e
        return (dim, mean, cov)

    def _process_parameters_psd(self, dim, mean, cov):
        if dim is None:
            if mean is None:
                if cov is None:
                    dim = 1
                else:
                    cov = np.asarray(cov, dtype=float)
                    if cov.ndim < 2:
                        dim = 1
                    else:
                        dim = cov.shape[0]
            else:
                mean = np.asarray(mean, dtype=float)
                dim = mean.size
        elif not np.isscalar(dim):
            raise ValueError('Dimension of random variable must be a scalar.')
        if mean is None:
            mean = np.zeros(dim)
        mean = np.asarray(mean, dtype=float)
        if cov is None:
            cov = 1.0
        cov = np.asarray(cov, dtype=float)
        if dim == 1:
            mean = mean.reshape(1)
            cov = cov.reshape(1, 1)
        if mean.ndim != 1 or mean.shape[0] != dim:
            raise ValueError("Array 'mean' must be a vector of length %d." % dim)
        if cov.ndim == 0:
            cov = cov * np.eye(dim)
        elif cov.ndim == 1:
            cov = np.diag(cov)
        elif cov.ndim == 2 and cov.shape != (dim, dim):
            rows, cols = cov.shape
            if rows != cols:
                msg = "Array 'cov' must be square if it is two dimensional, but cov.shape = %s." % str(cov.shape)
            else:
                msg = "Dimension mismatch: array 'cov' is of shape %s, but 'mean' is a vector of length %d."
                msg = msg % (str(cov.shape), len(mean))
            raise ValueError(msg)
        elif cov.ndim > 2:
            raise ValueError("Array 'cov' must be at most two-dimensional, but cov.ndim = %d" % cov.ndim)
        return (dim, mean, cov)

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

    def _logpdf(self, x, mean, cov_object):
        """Log of the multivariate normal probability density function.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function
        mean : ndarray
            Mean of the distribution
        cov_object : Covariance
            An object representing the Covariance matrix

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.

        """
        log_det_cov, rank = (cov_object.log_pdet, cov_object.rank)
        dev = x - mean
        if dev.ndim > 1:
            log_det_cov = log_det_cov[..., np.newaxis]
            rank = rank[..., np.newaxis]
        maha = np.sum(np.square(cov_object.whiten(dev)), axis=-1)
        return -0.5 * (rank * _LOG_2PI + log_det_cov + maha)

    def logpdf(self, x, mean=None, cov=1, allow_singular=False):
        """Log of the multivariate normal probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvn_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray or scalar
            Log of the probability density function evaluated at `x`

        Notes
        -----
        %(_mvn_doc_callparams_note)s

        """
        params = self._process_parameters(mean, cov, allow_singular)
        dim, mean, cov_object = params
        x = self._process_quantiles(x, dim)
        out = self._logpdf(x, mean, cov_object)
        if np.any(cov_object.rank < dim):
            out_of_bounds = ~cov_object._support_mask(x - mean)
            out[out_of_bounds] = -np.inf
        return _squeeze_output(out)

    def pdf(self, x, mean=None, cov=1, allow_singular=False):
        """Multivariate normal probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvn_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray or scalar
            Probability density function evaluated at `x`

        Notes
        -----
        %(_mvn_doc_callparams_note)s

        """
        params = self._process_parameters(mean, cov, allow_singular)
        dim, mean, cov_object = params
        x = self._process_quantiles(x, dim)
        out = np.exp(self._logpdf(x, mean, cov_object))
        if np.any(cov_object.rank < dim):
            out_of_bounds = ~cov_object._support_mask(x - mean)
            out[out_of_bounds] = 0.0
        return _squeeze_output(out)

    def _cdf(self, x, mean, cov, maxpts, abseps, releps, lower_limit):
        """Multivariate normal cumulative distribution function.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the cumulative distribution function.
        mean : ndarray
            Mean of the distribution
        cov : array_like
            Covariance matrix of the distribution
        maxpts : integer
            The maximum number of points to use for integration
        abseps : float
            Absolute error tolerance
        releps : float
            Relative error tolerance
        lower_limit : array_like, optional
            Lower limit of integration of the cumulative distribution function.
            Default is negative infinity. Must be broadcastable with `x`.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'cdf' instead.


        .. versionadded:: 1.0.0

        """
        lower = np.full(mean.shape, -np.inf) if lower_limit is None else lower_limit
        b, a = np.broadcast_arrays(x, lower)
        i_swap = b < a
        signs = (-1) ** i_swap.sum(axis=-1)
        a, b = (a.copy(), b.copy())
        a[i_swap], b[i_swap] = (b[i_swap], a[i_swap])
        n = x.shape[-1]
        limits = np.concatenate((a, b), axis=-1)

        def func1d(limits):
            return _mvn.mvnun(limits[:n], limits[n:], mean, cov, maxpts, abseps, releps)[0]
        out = np.apply_along_axis(func1d, -1, limits) * signs
        return _squeeze_output(out)

    def logcdf(self, x, mean=None, cov=1, allow_singular=False, maxpts=None, abseps=1e-05, releps=1e-05, *, lower_limit=None):
        """Log of the multivariate normal cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvn_doc_default_callparams)s
        maxpts : integer, optional
            The maximum number of points to use for integration
            (default `1000000*dim`)
        abseps : float, optional
            Absolute error tolerance (default 1e-5)
        releps : float, optional
            Relative error tolerance (default 1e-5)
        lower_limit : array_like, optional
            Lower limit of integration of the cumulative distribution function.
            Default is negative infinity. Must be broadcastable with `x`.

        Returns
        -------
        cdf : ndarray or scalar
            Log of the cumulative distribution function evaluated at `x`

        Notes
        -----
        %(_mvn_doc_callparams_note)s

        .. versionadded:: 1.0.0

        """
        params = self._process_parameters(mean, cov, allow_singular)
        dim, mean, cov_object = params
        cov = cov_object.covariance
        x = self._process_quantiles(x, dim)
        if not maxpts:
            maxpts = 1000000 * dim
        cdf = self._cdf(x, mean, cov, maxpts, abseps, releps, lower_limit)
        cdf = cdf + 0j if np.any(cdf < 0) else cdf
        out = np.log(cdf)
        return out

    def cdf(self, x, mean=None, cov=1, allow_singular=False, maxpts=None, abseps=1e-05, releps=1e-05, *, lower_limit=None):
        """Multivariate normal cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvn_doc_default_callparams)s
        maxpts : integer, optional
            The maximum number of points to use for integration
            (default `1000000*dim`)
        abseps : float, optional
            Absolute error tolerance (default 1e-5)
        releps : float, optional
            Relative error tolerance (default 1e-5)
        lower_limit : array_like, optional
            Lower limit of integration of the cumulative distribution function.
            Default is negative infinity. Must be broadcastable with `x`.

        Returns
        -------
        cdf : ndarray or scalar
            Cumulative distribution function evaluated at `x`

        Notes
        -----
        %(_mvn_doc_callparams_note)s

        .. versionadded:: 1.0.0

        """
        params = self._process_parameters(mean, cov, allow_singular)
        dim, mean, cov_object = params
        cov = cov_object.covariance
        x = self._process_quantiles(x, dim)
        if not maxpts:
            maxpts = 1000000 * dim
        out = self._cdf(x, mean, cov, maxpts, abseps, releps, lower_limit)
        return out

    def rvs(self, mean=None, cov=1, size=1, random_state=None):
        """Draw random samples from a multivariate normal distribution.

        Parameters
        ----------
        %(_mvn_doc_default_callparams)s
        size : integer, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.

        Notes
        -----
        %(_mvn_doc_callparams_note)s

        """
        dim, mean, cov_object = self._process_parameters(mean, cov)
        random_state = self._get_random_state(random_state)
        if isinstance(cov_object, _covariance.CovViaPSD):
            cov = cov_object.covariance
            out = random_state.multivariate_normal(mean, cov, size)
            out = _squeeze_output(out)
        else:
            size = size or tuple()
            if not np.iterable(size):
                size = (size,)
            shape = tuple(size) + (cov_object.shape[-1],)
            x = random_state.normal(size=shape)
            out = mean + cov_object.colorize(x)
        return out

    def entropy(self, mean=None, cov=1):
        """Compute the differential entropy of the multivariate normal.

        Parameters
        ----------
        %(_mvn_doc_default_callparams)s

        Returns
        -------
        h : scalar
            Entropy of the multivariate normal distribution

        Notes
        -----
        %(_mvn_doc_callparams_note)s

        """
        dim, mean, cov_object = self._process_parameters(mean, cov)
        return 0.5 * (cov_object.rank * (_LOG_2PI + 1) + cov_object.log_pdet)

    def fit(self, x, fix_mean=None, fix_cov=None):
        """Fit a multivariate normal distribution to data.

        Parameters
        ----------
        x : ndarray (m, n)
            Data the distribution is fitted to. Must have two axes.
            The first axis of length `m` represents the number of vectors
            the distribution is fitted to. The second axis of length `n`
            determines the dimensionality of the fitted distribution.
        fix_mean : ndarray(n, )
            Fixed mean vector. Must have length `n`.
        fix_cov: ndarray (n, n)
            Fixed covariance matrix. Must have shape `(n, n)`.

        Returns
        -------
        mean : ndarray (n, )
            Maximum likelihood estimate of the mean vector
        cov : ndarray (n, n)
            Maximum likelihood estimate of the covariance matrix

        """
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError('`x` must be two-dimensional.')
        n_vectors, dim = x.shape
        if fix_mean is not None:
            fix_mean = np.atleast_1d(fix_mean)
            if fix_mean.shape != (dim,):
                msg = '`fix_mean` must be a one-dimensional array the same length as the dimensionality of the vectors `x`.'
                raise ValueError(msg)
            mean = fix_mean
        else:
            mean = x.mean(axis=0)
        if fix_cov is not None:
            fix_cov = np.atleast_2d(fix_cov)
            if fix_cov.shape != (dim, dim):
                msg = '`fix_cov` must be a two-dimensional square array of same side length as the dimensionality of the vectors `x`.'
                raise ValueError(msg)
            s, u = scipy.linalg.eigh(fix_cov, lower=True, check_finite=True)
            eps = _eigvalsh_to_eps(s)
            if np.min(s) < -eps:
                msg = '`fix_cov` must be symmetric positive semidefinite.'
                raise ValueError(msg)
            cov = fix_cov
        else:
            centered_data = x - mean
            cov = centered_data.T @ centered_data / n_vectors
        return (mean, cov)