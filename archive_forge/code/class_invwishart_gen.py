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
class invwishart_gen(wishart_gen):
    """An inverse Wishart random variable.

    The `df` keyword specifies the degrees of freedom. The `scale` keyword
    specifies the scale matrix, which must be symmetric and positive definite.
    In this context, the scale matrix is often interpreted in terms of a
    multivariate normal covariance matrix.

    Methods
    -------
    pdf(x, df, scale)
        Probability density function.
    logpdf(x, df, scale)
        Log of the probability density function.
    rvs(df, scale, size=1, random_state=None)
        Draw random samples from an inverse Wishart distribution.
    entropy(df, scale)
        Differential entropy of the distribution.

    Parameters
    ----------
    %(_doc_default_callparams)s
    %(_doc_random_state)s

    Raises
    ------
    scipy.linalg.LinAlgError
        If the scale matrix `scale` is not positive definite.

    See Also
    --------
    wishart

    Notes
    -----
    %(_doc_callparams_note)s

    The scale matrix `scale` must be a symmetric positive definite
    matrix. Singular matrices, including the symmetric positive semi-definite
    case, are not supported. Symmetry is not checked; only the lower triangular
    portion is used.

    The inverse Wishart distribution is often denoted

    .. math::

        W_p^{-1}(\\nu, \\Psi)

    where :math:`\\nu` is the degrees of freedom and :math:`\\Psi` is the
    :math:`p \\times p` scale matrix.

    The probability density function for `invwishart` has support over positive
    definite matrices :math:`S`; if :math:`S \\sim W^{-1}_p(\\nu, \\Sigma)`,
    then its PDF is given by:

    .. math::

        f(S) = \\frac{|\\Sigma|^\\frac{\\nu}{2}}{2^{ \\frac{\\nu p}{2} }
               |S|^{\\frac{\\nu + p + 1}{2}} \\Gamma_p \\left(\\frac{\\nu}{2} \\right)}
               \\exp\\left( -tr(\\Sigma S^{-1}) / 2 \\right)

    If :math:`S \\sim W_p^{-1}(\\nu, \\Psi)` (inverse Wishart) then
    :math:`S^{-1} \\sim W_p(\\nu, \\Psi^{-1})` (Wishart).

    If the scale matrix is 1-dimensional and equal to one, then the inverse
    Wishart distribution :math:`W_1(\\nu, 1)` collapses to the
    inverse Gamma distribution with parameters shape = :math:`\\frac{\\nu}{2}`
    and scale = :math:`\\frac{1}{2}`.

    Instead of inverting a randomly generated Wishart matrix as described in [2],
    here the algorithm in [4] is used to directly generate a random inverse-Wishart
    matrix without inversion.

    .. versionadded:: 0.16.0

    References
    ----------
    .. [1] M.L. Eaton, "Multivariate Statistics: A Vector Space Approach",
           Wiley, 1983.
    .. [2] M.C. Jones, "Generating Inverse Wishart Matrices", Communications
           in Statistics - Simulation and Computation, vol. 14.2, pp.511-514,
           1985.
    .. [3] Gupta, M. and Srivastava, S. "Parametric Bayesian Estimation of
           Differential Entropy and Relative Entropy". Entropy 12, 818 - 843.
           2010.
    .. [4] S.D. Axen, "Efficiently generating inverse-Wishart matrices and
           their Cholesky factors", :arXiv:`2310.15884v1`. 2023.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import invwishart, invgamma
    >>> x = np.linspace(0.01, 1, 100)
    >>> iw = invwishart.pdf(x, df=6, scale=1)
    >>> iw[:3]
    array([  1.20546865e-15,   5.42497807e-06,   4.45813929e-03])
    >>> ig = invgamma.pdf(x, 6/2., scale=1./2)
    >>> ig[:3]
    array([  1.20546865e-15,   5.42497807e-06,   4.45813929e-03])
    >>> plt.plot(x, iw)
    >>> plt.show()

    The input quantiles can be any shape of array, as long as the last
    axis labels the components.

    Alternatively, the object may be called (as a function) to fix the degrees
    of freedom and scale parameters, returning a "frozen" inverse Wishart
    random variable:

    >>> rv = invwishart(df=1, scale=1)
    >>> # Frozen object with the same methods but holding the given
    >>> # degrees of freedom and scale fixed.

    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__, wishart_docdict_params)

    def __call__(self, df=None, scale=None, seed=None):
        """Create a frozen inverse Wishart distribution.

        See `invwishart_frozen` for more information.

        """
        return invwishart_frozen(df, scale, seed)

    def _logpdf(self, x, dim, df, log_det_scale, C):
        """Log of the inverse Wishart probability density function.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function.
        dim : int
            Dimension of the scale matrix
        df : int
            Degrees of freedom
        log_det_scale : float
            Logarithm of the determinant of the scale matrix
        C : ndarray
            Cholesky factorization of the scale matrix, lower triagular.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.

        """
        log_det_x = np.empty(x.shape[-1])
        tr_scale_x_inv = np.empty(x.shape[-1])
        trsm = get_blas_funcs('trsm', (x,))
        if dim > 1:
            for i in range(x.shape[-1]):
                Cx, log_det_x[i] = self._cholesky_logdet(x[:, :, i])
                A = trsm(1.0, Cx, C, side=0, lower=True)
                tr_scale_x_inv[i] = np.linalg.norm(A) ** 2
        else:
            log_det_x[:] = np.log(x[0, 0])
            tr_scale_x_inv[:] = C[0, 0] ** 2 / x[0, 0]
        out = 0.5 * df * log_det_scale - 0.5 * tr_scale_x_inv - (0.5 * df * dim * _LOG_2 + 0.5 * (df + dim + 1) * log_det_x) - multigammaln(0.5 * df, dim)
        return out

    def logpdf(self, x, df, scale):
        """Log of the inverse Wishart probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
            Each quantile must be a symmetric positive definite matrix.
        %(_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray
            Log of the probability density function evaluated at `x`

        Notes
        -----
        %(_doc_callparams_note)s

        """
        dim, df, scale = self._process_parameters(df, scale)
        x = self._process_quantiles(x, dim)
        C, log_det_scale = self._cholesky_logdet(scale)
        out = self._logpdf(x, dim, df, log_det_scale, C)
        return _squeeze_output(out)

    def pdf(self, x, df, scale):
        """Inverse Wishart probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
            Each quantile must be a symmetric positive definite matrix.
        %(_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at `x`

        Notes
        -----
        %(_doc_callparams_note)s

        """
        return np.exp(self.logpdf(x, df, scale))

    def _mean(self, dim, df, scale):
        """Mean of the inverse Wishart distribution.

        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
        %(_doc_default_callparams)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'mean' instead.

        """
        if df > dim + 1:
            out = scale / (df - dim - 1)
        else:
            out = None
        return out

    def mean(self, df, scale):
        """Mean of the inverse Wishart distribution.

        Only valid if the degrees of freedom are greater than the dimension of
        the scale matrix plus one.

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        mean : float or None
            The mean of the distribution

        """
        dim, df, scale = self._process_parameters(df, scale)
        out = self._mean(dim, df, scale)
        return _squeeze_output(out) if out is not None else out

    def _mode(self, dim, df, scale):
        """Mode of the inverse Wishart distribution.

        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
        %(_doc_default_callparams)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'mode' instead.

        """
        return scale / (df + dim + 1)

    def mode(self, df, scale):
        """Mode of the inverse Wishart distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        mode : float
            The Mode of the distribution

        """
        dim, df, scale = self._process_parameters(df, scale)
        out = self._mode(dim, df, scale)
        return _squeeze_output(out)

    def _var(self, dim, df, scale):
        """Variance of the inverse Wishart distribution.

        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
        %(_doc_default_callparams)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'var' instead.

        """
        if df > dim + 3:
            var = (df - dim + 1) * scale ** 2
            diag = scale.diagonal()
            var += (df - dim - 1) * np.outer(diag, diag)
            var /= (df - dim) * (df - dim - 1) ** 2 * (df - dim - 3)
        else:
            var = None
        return var

    def var(self, df, scale):
        """Variance of the inverse Wishart distribution.

        Only valid if the degrees of freedom are greater than the dimension of
        the scale matrix plus three.

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        var : float
            The variance of the distribution
        """
        dim, df, scale = self._process_parameters(df, scale)
        out = self._var(dim, df, scale)
        return _squeeze_output(out) if out is not None else out

    def _inv_standard_rvs(self, n, shape, dim, df, random_state):
        """
        Parameters
        ----------
        n : integer
            Number of variates to generate
        shape : iterable
            Shape of the variates to generate
        dim : int
            Dimension of the scale matrix
        df : int
            Degrees of freedom
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
        A : ndarray
            Random variates of shape (`shape`) + (``dim``, ``dim``).
            Each slice `A[..., :, :]` is lower-triangular, and its
            inverse is the lower Cholesky factor of a draw from
            `invwishart(df, np.eye(dim))`.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'rvs' instead.

        """
        A = np.zeros(shape + (dim, dim))
        tri_rows, tri_cols = np.tril_indices(dim, k=-1)
        n_tril = dim * (dim - 1) // 2
        A[..., tri_rows, tri_cols] = random_state.normal(size=(*shape, n_tril))
        rows = np.arange(dim)
        chi_dfs = df - dim + 1 + rows
        A[..., rows, rows] = random_state.chisquare(df=chi_dfs, size=(*shape, dim)) ** 0.5
        return A

    def _rvs(self, n, shape, dim, df, C, random_state):
        """Draw random samples from an inverse Wishart distribution.

        Parameters
        ----------
        n : integer
            Number of variates to generate
        shape : iterable
            Shape of the variates to generate
        dim : int
            Dimension of the scale matrix
        df : int
            Degrees of freedom
        C : ndarray
            Cholesky factorization of the scale matrix, lower triagular.
        %(_doc_random_state)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'rvs' instead.

        """
        random_state = self._get_random_state(random_state)
        A = self._inv_standard_rvs(n, shape, dim, df, random_state)
        trsm = get_blas_funcs('trsm', (A,))
        trmm = get_blas_funcs('trmm', (A,))
        for index in np.ndindex(A.shape[:-2]):
            if dim > 1:
                CA = trsm(1.0, A[index], C, side=1, lower=True)
                A[index] = trmm(1.0, CA, CA, side=1, lower=True, trans_a=True)
            else:
                A[index][0, 0] = (C[0, 0] / A[index][0, 0]) ** 2
        return A

    def rvs(self, df, scale, size=1, random_state=None):
        """Draw random samples from an inverse Wishart distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s
        size : integer or iterable of integers, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray
            Random variates of shape (`size`) + (``dim``, ``dim``), where
            ``dim`` is the dimension of the scale matrix.

        Notes
        -----
        %(_doc_callparams_note)s

        """
        n, shape = self._process_size(size)
        dim, df, scale = self._process_parameters(df, scale)
        C = scipy.linalg.cholesky(scale, lower=True)
        out = self._rvs(n, shape, dim, df, C, random_state)
        return _squeeze_output(out)

    def _entropy(self, dim, df, log_det_scale):
        psi_eval_points = [0.5 * (df - dim + i) for i in range(1, dim + 1)]
        psi_eval_points = np.asarray(psi_eval_points)
        return multigammaln(0.5 * df, dim) + 0.5 * dim * df + 0.5 * (dim + 1) * (log_det_scale - _LOG_2) - 0.5 * (df + dim + 1) * psi(psi_eval_points, out=psi_eval_points).sum()

    def entropy(self, df, scale):
        dim, df, scale = self._process_parameters(df, scale)
        _, log_det_scale = self._cholesky_logdet(scale)
        return self._entropy(dim, df, log_det_scale)