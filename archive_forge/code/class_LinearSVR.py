import warnings
from numbers import Integral, Real
import numpy as np
from ..base import BaseEstimator, OutlierMixin, RegressorMixin, _fit_context
from ..linear_model._base import LinearClassifierMixin, LinearModel, SparseCoefMixin
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.multiclass import check_classification_targets
from ..utils.validation import _num_samples
from ._base import BaseLibSVM, BaseSVC, _fit_liblinear, _get_liblinear_solver_type
class LinearSVR(RegressorMixin, LinearModel):
    """Linear Support Vector Regression.

    Similar to SVR with parameter kernel='linear', but implemented in terms of
    liblinear rather than libsvm, so it has more flexibility in the choice of
    penalties and loss functions and should scale better to large numbers of
    samples.

    The main differences between :class:`~sklearn.svm.LinearSVR` and
    :class:`~sklearn.svm.SVR` lie in the loss function used by default, and in
    the handling of intercept regularization between those two implementations.

    This class supports both dense and sparse input.

    Read more in the :ref:`User Guide <svm_regression>`.

    .. versionadded:: 0.16

    Parameters
    ----------
    epsilon : float, default=0.0
        Epsilon parameter in the epsilon-insensitive loss function. Note
        that the value of this parameter depends on the scale of the target
        variable y. If unsure, set ``epsilon=0``.

    tol : float, default=1e-4
        Tolerance for stopping criteria.

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.

    loss : {'epsilon_insensitive', 'squared_epsilon_insensitive'},             default='epsilon_insensitive'
        Specifies the loss function. The epsilon-insensitive loss
        (standard SVR) is the L1 loss, while the squared epsilon-insensitive
        loss ('squared_epsilon_insensitive') is the L2 loss.

    fit_intercept : bool, default=True
        Whether or not to fit an intercept. If set to True, the feature vector
        is extended to include an intercept term: `[x_1, ..., x_n, 1]`, where
        1 corresponds to the intercept. If set to False, no intercept will be
        used in calculations (i.e. data is expected to be already centered).

    intercept_scaling : float, default=1.0
        When `fit_intercept` is True, the instance vector x becomes `[x_1, ...,
        x_n, intercept_scaling]`, i.e. a "synthetic" feature with a constant
        value equal to `intercept_scaling` is appended to the instance vector.
        The intercept becomes intercept_scaling * synthetic feature weight.
        Note that liblinear internally penalizes the intercept, treating it
        like any other term in the feature vector. To reduce the impact of the
        regularization on the intercept, the `intercept_scaling` parameter can
        be set to a value greater than 1; the higher the value of
        `intercept_scaling`, the lower the impact of regularization on it.
        Then, the weights become `[w_x_1, ..., w_x_n,
        w_intercept*intercept_scaling]`, where `w_x_1, ..., w_x_n` represent
        the feature weights and the intercept weight is scaled by
        `intercept_scaling`. This scaling allows the intercept term to have a
        different regularization behavior compared to the other features.

    dual : "auto" or bool, default=True
        Select the algorithm to either solve the dual or primal
        optimization problem. Prefer dual=False when n_samples > n_features.
        `dual="auto"` will choose the value of the parameter automatically,
        based on the values of `n_samples`, `n_features` and `loss`. If
        `n_samples` < `n_features` and optimizer supports chosen `loss`,
        then dual will be set to True, otherwise it will be set to False.

        .. versionchanged:: 1.3
           The `"auto"` option is added in version 1.3 and will be the default
           in version 1.5.

    verbose : int, default=0
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in liblinear that, if enabled, may not work
        properly in a multithreaded context.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for shuffling the data.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    max_iter : int, default=1000
        The maximum number of iterations to be run.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features) if n_classes == 2             else (n_classes, n_features)
        Weights assigned to the features (coefficients in the primal
        problem).

        `coef_` is a readonly property derived from `raw_coef_` that
        follows the internal memory layout of liblinear.

    intercept_ : ndarray of shape (1) if n_classes == 2 else (n_classes)
        Constants in decision function.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        Maximum number of iterations run across all classes.

    See Also
    --------
    LinearSVC : Implementation of Support Vector Machine classifier using the
        same library as this class (liblinear).

    SVR : Implementation of Support Vector Machine regression using libsvm:
        the kernel can be non-linear but its SMO algorithm does not scale to
        large number of samples as :class:`~sklearn.svm.LinearSVR` does.

    sklearn.linear_model.SGDRegressor : SGDRegressor can optimize the same cost
        function as LinearSVR
        by adjusting the penalty and loss parameters. In addition it requires
        less memory, allows incremental (online) learning, and implements
        various loss functions and regularization regimes.

    Examples
    --------
    >>> from sklearn.svm import LinearSVR
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=4, random_state=0)
    >>> regr = make_pipeline(StandardScaler(),
    ...                      LinearSVR(dual="auto", random_state=0, tol=1e-5))
    >>> regr.fit(X, y)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('linearsvr', LinearSVR(dual='auto', random_state=0, tol=1e-05))])

    >>> print(regr.named_steps['linearsvr'].coef_)
    [18.582... 27.023... 44.357... 64.522...]
    >>> print(regr.named_steps['linearsvr'].intercept_)
    [-4...]
    >>> print(regr.predict([[0, 0, 0, 0]]))
    [-2.384...]
    """
    _parameter_constraints: dict = {'epsilon': [Real], 'tol': [Interval(Real, 0.0, None, closed='neither')], 'C': [Interval(Real, 0.0, None, closed='neither')], 'loss': [StrOptions({'epsilon_insensitive', 'squared_epsilon_insensitive'})], 'fit_intercept': ['boolean'], 'intercept_scaling': [Interval(Real, 0, None, closed='neither')], 'dual': ['boolean', StrOptions({'auto'}), Hidden(StrOptions({'warn'}))], 'verbose': ['verbose'], 'random_state': ['random_state'], 'max_iter': [Interval(Integral, 0, None, closed='left')]}

    def __init__(self, *, epsilon=0.0, tol=0.0001, C=1.0, loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1.0, dual='warn', verbose=0, random_state=None, max_iter=1000):
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter
        self.dual = dual
        self.loss = loss

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual
            samples. If not provided,
            then each sample is given unit weight.

            .. versionadded:: 0.18

        Returns
        -------
        self : object
            An instance of the estimator.
        """
        X, y = self._validate_data(X, y, accept_sparse='csr', dtype=np.float64, order='C', accept_large_sparse=False)
        penalty = 'l2'
        _dual = _validate_dual_parameter(self.dual, self.loss, penalty, 'ovr', X)
        self.coef_, self.intercept_, n_iter_ = _fit_liblinear(X, y, self.C, self.fit_intercept, self.intercept_scaling, None, penalty, _dual, self.verbose, self.max_iter, self.tol, self.random_state, loss=self.loss, epsilon=self.epsilon, sample_weight=sample_weight)
        self.coef_ = self.coef_.ravel()
        self.n_iter_ = n_iter_.max().item()
        return self

    def _more_tags(self):
        return {'_xfail_checks': {'check_sample_weights_invariance': 'zero sample_weight is not equivalent to removing samples'}}