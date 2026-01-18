import sys
import warnings
from math import log
from numbers import Integral, Real
import numpy as np
from scipy import interpolate, linalg
from scipy.linalg.lapack import get_lapack_funcs
from ..base import MultiOutputMixin, RegressorMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..model_selection import check_cv
from ..utils import (  # type: ignore
from ..utils._metadata_requests import (
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.parallel import Parallel, delayed
from ._base import LinearModel, LinearRegression, _preprocess_data
class LassoLarsIC(LassoLars):
    """Lasso model fit with Lars using BIC or AIC for model selection.

    The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    AIC is the Akaike information criterion [2]_ and BIC is the Bayes
    Information criterion [3]_. Such criteria are useful to select the value
    of the regularization parameter by making a trade-off between the
    goodness of fit and the complexity of the model. A good model should
    explain well the data while being simple.

    Read more in the :ref:`User Guide <lasso_lars_ic>`.

    Parameters
    ----------
    criterion : {'aic', 'bic'}, default='aic'
        The type of criterion to use.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    verbose : bool or int, default=False
        Sets the verbosity amount.

    precompute : bool, 'auto' or array-like, default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    max_iter : int, default=500
        Maximum number of iterations to perform. Can be used for
        early stopping.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    positive : bool, default=False
        Restrict coefficients to be >= 0. Be aware that you might want to
        remove fit_intercept which is set True by default.
        Under the positive restriction the model coefficients do not converge
        to the ordinary-least-squares solution for small values of alpha.
        Only coefficients up to the smallest alpha value (``alphas_[alphas_ >
        0.].min()`` when fit_path=True) reached by the stepwise Lars-Lasso
        algorithm are typically in congruence with the solution of the
        coordinate descent Lasso estimator.
        As a consequence using LassoLarsIC only makes sense for problems where
        a sparse solution is expected and/or reached.

    noise_variance : float, default=None
        The estimated noise variance of the data. If `None`, an unbiased
        estimate is computed by an OLS model. However, it is only possible
        in the case where `n_samples > n_features + fit_intercept`.

        .. versionadded:: 1.1

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        parameter vector (w in the formulation formula)

    intercept_ : float
        independent term in decision function.

    alpha_ : float
        the alpha parameter chosen by the information criterion

    alphas_ : array-like of shape (n_alphas + 1,) or list of such arrays
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter``, ``n_features`` or the
        number of nodes in the path with ``alpha >= alpha_min``, whichever
        is smaller. If a list, it will be of length `n_targets`.

    n_iter_ : int
        number of iterations run by lars_path to find the grid of
        alphas.

    criterion_ : array-like of shape (n_alphas,)
        The value of the information criteria ('aic', 'bic') across all
        alphas. The alpha which has the smallest information criterion is
        chosen, as specified in [1]_.

    noise_variance_ : float
        The estimated noise variance from the data used to compute the
        criterion.

        .. versionadded:: 1.1

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    lars_path : Compute Least Angle Regression or Lasso
        path using LARS algorithm.
    lasso_path : Compute Lasso path with coordinate descent.
    Lasso : Linear Model trained with L1 prior as
        regularizer (aka the Lasso).
    LassoCV : Lasso linear model with iterative fitting
        along a regularization path.
    LassoLars : Lasso model fit with Least Angle Regression a.k.a. Lars.
    LassoLarsCV: Cross-validated Lasso, using the LARS algorithm.
    sklearn.decomposition.sparse_encode : Sparse coding.

    Notes
    -----
    The number of degrees of freedom is computed as in [1]_.

    To have more details regarding the mathematical formulation of the
    AIC and BIC criteria, please refer to :ref:`User Guide <lasso_lars_ic>`.

    References
    ----------
    .. [1] :arxiv:`Zou, Hui, Trevor Hastie, and Robert Tibshirani.
            "On the degrees of freedom of the lasso."
            The Annals of Statistics 35.5 (2007): 2173-2192.
            <0712.0881>`

    .. [2] `Wikipedia entry on the Akaike information criterion
            <https://en.wikipedia.org/wiki/Akaike_information_criterion>`_

    .. [3] `Wikipedia entry on the Bayesian information criterion
            <https://en.wikipedia.org/wiki/Bayesian_information_criterion>`_

    Examples
    --------
    >>> from sklearn import linear_model
    >>> reg = linear_model.LassoLarsIC(criterion='bic')
    >>> X = [[-2, 2], [-1, 1], [0, 0], [1, 1], [2, 2]]
    >>> y = [-2.2222, -1.1111, 0, -1.1111, -2.2222]
    >>> reg.fit(X, y)
    LassoLarsIC(criterion='bic')
    >>> print(reg.coef_)
    [ 0.  -1.11...]
    """
    _parameter_constraints: dict = {**LassoLars._parameter_constraints, 'criterion': [StrOptions({'aic', 'bic'})], 'noise_variance': [Interval(Real, 0, None, closed='left'), None]}
    for parameter in ['jitter', 'fit_path', 'alpha', 'random_state']:
        _parameter_constraints.pop(parameter)

    def __init__(self, criterion='aic', *, fit_intercept=True, verbose=False, precompute='auto', max_iter=500, eps=np.finfo(float).eps, copy_X=True, positive=False, noise_variance=None):
        self.criterion = criterion
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.max_iter = max_iter
        self.verbose = verbose
        self.copy_X = copy_X
        self.precompute = precompute
        self.eps = eps
        self.fit_path = True
        self.noise_variance = noise_variance

    def _more_tags(self):
        return {'multioutput': False}

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, copy_X=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary.

        copy_X : bool, default=None
            If provided, this parameter will override the choice
            of copy_X made at instance creation.
            If ``True``, X will be copied; else, it may be overwritten.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        if copy_X is None:
            copy_X = self.copy_X
        X, y = self._validate_data(X, y, y_numeric=True)
        X, y, Xmean, ymean, Xstd = _preprocess_data(X, y, fit_intercept=self.fit_intercept, copy=copy_X)
        Gram = self.precompute
        alphas_, _, coef_path_, self.n_iter_ = lars_path(X, y, Gram=Gram, copy_X=copy_X, copy_Gram=True, alpha_min=0.0, method='lasso', verbose=self.verbose, max_iter=self.max_iter, eps=self.eps, return_n_iter=True, positive=self.positive)
        n_samples = X.shape[0]
        if self.criterion == 'aic':
            criterion_factor = 2
        elif self.criterion == 'bic':
            criterion_factor = log(n_samples)
        else:
            raise ValueError(f'criterion should be either bic or aic, got {self.criterion!r}')
        residuals = y[:, np.newaxis] - np.dot(X, coef_path_)
        residuals_sum_squares = np.sum(residuals ** 2, axis=0)
        degrees_of_freedom = np.zeros(coef_path_.shape[1], dtype=int)
        for k, coef in enumerate(coef_path_.T):
            mask = np.abs(coef) > np.finfo(coef.dtype).eps
            if not np.any(mask):
                continue
            degrees_of_freedom[k] = np.sum(mask)
        self.alphas_ = alphas_
        if self.noise_variance is None:
            self.noise_variance_ = self._estimate_noise_variance(X, y, positive=self.positive)
        else:
            self.noise_variance_ = self.noise_variance
        self.criterion_ = n_samples * np.log(2 * np.pi * self.noise_variance_) + residuals_sum_squares / self.noise_variance_ + criterion_factor * degrees_of_freedom
        n_best = np.argmin(self.criterion_)
        self.alpha_ = alphas_[n_best]
        self.coef_ = coef_path_[:, n_best]
        self._set_intercept(Xmean, ymean, Xstd)
        return self

    def _estimate_noise_variance(self, X, y, positive):
        """Compute an estimate of the variance with an OLS model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to be fitted by the OLS model. We expect the data to be
            centered.

        y : ndarray of shape (n_samples,)
            Associated target.

        positive : bool, default=False
            Restrict coefficients to be >= 0. This should be inline with
            the `positive` parameter from `LassoLarsIC`.

        Returns
        -------
        noise_variance : float
            An estimator of the noise variance of an OLS model.
        """
        if X.shape[0] <= X.shape[1] + self.fit_intercept:
            raise ValueError(f'You are using {self.__class__.__name__} in the case where the number of samples is smaller than the number of features. In this setting, getting a good estimate for the variance of the noise is not possible. Provide an estimate of the noise variance in the constructor.')
        ols_model = LinearRegression(positive=positive, fit_intercept=False)
        y_pred = ols_model.fit(X, y).predict(X)
        return np.sum((y - y_pred) ** 2) / (X.shape[0] - X.shape[1] - self.fit_intercept)