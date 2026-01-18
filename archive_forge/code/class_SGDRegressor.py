import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from ..base import (
from ..exceptions import ConvergenceWarning
from ..model_selection import ShuffleSplit, StratifiedShuffleSplit
from ..utils import check_random_state, compute_class_weight, deprecated
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.metaestimators import available_if
from ..utils.multiclass import _check_partial_fit_first_call
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._base import LinearClassifierMixin, SparseCoefMixin, make_dataset
from ._sgd_fast import (
class SGDRegressor(BaseSGDRegressor):
    """Linear model fitted by minimizing a regularized empirical loss with SGD.

    SGD stands for Stochastic Gradient Descent: the gradient of the loss is
    estimated each sample at a time and the model is updated along the way with
    a decreasing strength schedule (aka learning rate).

    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector using either the squared euclidean norm
    L2 or the absolute norm L1 or a combination of both (Elastic Net). If the
    parameter update crosses the 0.0 value because of the regularizer, the
    update is truncated to 0.0 to allow for learning sparse models and achieve
    online feature selection.

    This implementation works with data represented as dense numpy arrays of
    floating point values for the features.

    Read more in the :ref:`User Guide <sgd>`.

    Parameters
    ----------
    loss : str, default='squared_error'
        The loss function to be used. The possible values are 'squared_error',
        'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'

        The 'squared_error' refers to the ordinary least squares fit.
        'huber' modifies 'squared_error' to focus less on getting outliers
        correct by switching from squared to linear loss past a distance of
        epsilon. 'epsilon_insensitive' ignores errors less than epsilon and is
        linear past that; this is the loss function used in SVR.
        'squared_epsilon_insensitive' is the same but becomes squared loss past
        a tolerance of epsilon.

        More details about the losses formulas can be found in the
        :ref:`User Guide <sgd_mathematical_formulation>`.

    penalty : {'l2', 'l1', 'elasticnet', None}, default='l2'
        The penalty (aka regularization term) to be used. Defaults to 'l2'
        which is the standard regularizer for linear SVM models. 'l1' and
        'elasticnet' might bring sparsity to the model (feature selection)
        not achievable with 'l2'. No penalty is added when set to `None`.

    alpha : float, default=0.0001
        Constant that multiplies the regularization term. The higher the
        value, the stronger the regularization. Also used to compute the
        learning rate when `learning_rate` is set to 'optimal'.
        Values must be in the range `[0.0, inf)`.

    l1_ratio : float, default=0.15
        The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
        Only used if `penalty` is 'elasticnet'.
        Values must be in the range `[0.0, 1.0]`.

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered.

    max_iter : int, default=1000
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        :meth:`partial_fit` method.
        Values must be in the range `[1, inf)`.

        .. versionadded:: 0.19

    tol : float or None, default=1e-3
        The stopping criterion. If it is not None, training will stop
        when (loss > best_loss - tol) for ``n_iter_no_change`` consecutive
        epochs.
        Convergence is checked against the training loss or the
        validation loss depending on the `early_stopping` parameter.
        Values must be in the range `[0.0, inf)`.

        .. versionadded:: 0.19

    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.

    verbose : int, default=0
        The verbosity level.
        Values must be in the range `[0, inf)`.

    epsilon : float, default=0.1
        Epsilon in the epsilon-insensitive loss functions; only if `loss` is
        'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.
        For 'huber', determines the threshold at which it becomes less
        important to get the prediction exactly right.
        For epsilon-insensitive, any differences between the current prediction
        and the correct label are ignored if they are less than this threshold.
        Values must be in the range `[0.0, inf)`.

    random_state : int, RandomState instance, default=None
        Used for shuffling the data, when ``shuffle`` is set to ``True``.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    learning_rate : str, default='invscaling'
        The learning rate schedule:

        - 'constant': `eta = eta0`
        - 'optimal': `eta = 1.0 / (alpha * (t + t0))`
          where t0 is chosen by a heuristic proposed by Leon Bottou.
        - 'invscaling': `eta = eta0 / pow(t, power_t)`
        - 'adaptive': eta = eta0, as long as the training keeps decreasing.
          Each time n_iter_no_change consecutive epochs fail to decrease the
          training loss by tol or fail to increase validation score by tol if
          early_stopping is True, the current learning rate is divided by 5.

            .. versionadded:: 0.20
                Added 'adaptive' option

    eta0 : float, default=0.01
        The initial learning rate for the 'constant', 'invscaling' or
        'adaptive' schedules. The default value is 0.01.
        Values must be in the range `[0.0, inf)`.

    power_t : float, default=0.25
        The exponent for inverse scaling learning rate.
        Values must be in the range `(-inf, inf)`.

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to True, it will automatically set aside
        a fraction of training data as validation and terminate
        training when validation score returned by the `score` method is not
        improving by at least `tol` for `n_iter_no_change` consecutive
        epochs.

        .. versionadded:: 0.20
            Added 'early_stopping' option

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if `early_stopping` is True.
        Values must be in the range `(0.0, 1.0)`.

        .. versionadded:: 0.20
            Added 'validation_fraction' option

    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before stopping
        fitting.
        Convergence is checked against the training loss or the
        validation loss depending on the `early_stopping` parameter.
        Integer values must be in the range `[1, max_iter)`.

        .. versionadded:: 0.20
            Added 'n_iter_no_change' option

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

        Repeatedly calling fit or partial_fit when warm_start is True can
        result in a different solution than when calling fit a single time
        because of the way the data is shuffled.
        If a dynamic learning rate is used, the learning rate is adapted
        depending on the number of samples already seen. Calling ``fit`` resets
        this counter, while ``partial_fit``  will result in increasing the
        existing counter.

    average : bool or int, default=False
        When set to True, computes the averaged SGD weights across all
        updates and stores the result in the ``coef_`` attribute. If set to
        an int greater than 1, averaging will begin once the total number of
        samples seen reaches `average`. So ``average=10`` will begin
        averaging after seeing 10 samples.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Weights assigned to the features.

    intercept_ : ndarray of shape (1,)
        The intercept term.

    n_iter_ : int
        The actual number of iterations before reaching the stopping criterion.

    t_ : int
        Number of weight updates performed during training.
        Same as ``(n_iter_ * n_samples + 1)``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    HuberRegressor : Linear regression model that is robust to outliers.
    Lars : Least Angle Regression model.
    Lasso : Linear Model trained with L1 prior as regularizer.
    RANSACRegressor : RANSAC (RANdom SAmple Consensus) algorithm.
    Ridge : Linear least squares with l2 regularization.
    sklearn.svm.SVR : Epsilon-Support Vector Regression.
    TheilSenRegressor : Theil-Sen Estimator robust multivariate regression model.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import SGDRegressor
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> n_samples, n_features = 10, 5
    >>> rng = np.random.RandomState(0)
    >>> y = rng.randn(n_samples)
    >>> X = rng.randn(n_samples, n_features)
    >>> # Always scale the input. The most convenient way is to use a pipeline.
    >>> reg = make_pipeline(StandardScaler(),
    ...                     SGDRegressor(max_iter=1000, tol=1e-3))
    >>> reg.fit(X, y)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('sgdregressor', SGDRegressor())])
    """
    _parameter_constraints: dict = {**BaseSGDRegressor._parameter_constraints, 'penalty': [StrOptions({'l2', 'l1', 'elasticnet'}), None], 'alpha': [Interval(Real, 0, None, closed='left')], 'l1_ratio': [Interval(Real, 0, 1, closed='both')], 'power_t': [Interval(Real, None, None, closed='neither')], 'learning_rate': [StrOptions({'constant', 'optimal', 'invscaling', 'adaptive'}), Hidden(StrOptions({'pa1', 'pa2'}))], 'epsilon': [Interval(Real, 0, None, closed='left')], 'eta0': [Interval(Real, 0, None, closed='left')]}

    def __init__(self, loss='squared_error', *, penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=DEFAULT_EPSILON, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False):
        super().__init__(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, shuffle=shuffle, verbose=verbose, epsilon=epsilon, random_state=random_state, learning_rate=learning_rate, eta0=eta0, power_t=power_t, early_stopping=early_stopping, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, warm_start=warm_start, average=average)

    def _more_tags(self):
        return {'_xfail_checks': {'check_sample_weights_invariance': 'zero sample_weight is not equivalent to removing samples'}, 'preserves_dtype': [np.float64, np.float32]}