import warnings
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from .base import (
from .utils import check_random_state
from .utils._param_validation import Interval, StrOptions
from .utils.multiclass import class_distribution
from .utils.random import _random_choice_csc
from .utils.stats import _weighted_percentile
from .utils.validation import (
class DummyRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Regressor that makes predictions using simple rules.

    This regressor is useful as a simple baseline to compare with other
    (real) regressors. Do not use it for real problems.

    Read more in the :ref:`User Guide <dummy_estimators>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    strategy : {"mean", "median", "quantile", "constant"}, default="mean"
        Strategy to use to generate predictions.

        * "mean": always predicts the mean of the training set
        * "median": always predicts the median of the training set
        * "quantile": always predicts a specified quantile of the training set,
          provided with the quantile parameter.
        * "constant": always predicts a constant value that is provided by
          the user.

    constant : int or float or array-like of shape (n_outputs,), default=None
        The explicit constant as predicted by the "constant" strategy. This
        parameter is useful only for the "constant" strategy.

    quantile : float in [0.0, 1.0], default=None
        The quantile to predict using the "quantile" strategy. A quantile of
        0.5 corresponds to the median, while 0.0 to the minimum and 1.0 to the
        maximum.

    Attributes
    ----------
    constant_ : ndarray of shape (1, n_outputs)
        Mean or median or quantile of the training targets or constant value
        given by the user.

    n_outputs_ : int
        Number of outputs.

    See Also
    --------
    DummyClassifier: Classifier that makes predictions using simple rules.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.dummy import DummyRegressor
    >>> X = np.array([1.0, 2.0, 3.0, 4.0])
    >>> y = np.array([2.0, 3.0, 5.0, 10.0])
    >>> dummy_regr = DummyRegressor(strategy="mean")
    >>> dummy_regr.fit(X, y)
    DummyRegressor()
    >>> dummy_regr.predict(X)
    array([5., 5., 5., 5.])
    >>> dummy_regr.score(X, y)
    0.0
    """
    _parameter_constraints: dict = {'strategy': [StrOptions({'mean', 'median', 'quantile', 'constant'})], 'quantile': [Interval(Real, 0.0, 1.0, closed='both'), None], 'constant': [Interval(Real, None, None, closed='neither'), 'array-like', None]}

    def __init__(self, *, strategy='mean', constant=None, quantile=None):
        self.strategy = strategy
        self.constant = constant
        self.quantile = quantile

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit the random regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        y = check_array(y, ensure_2d=False, input_name='y')
        if len(y) == 0:
            raise ValueError('y must not be empty.')
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
        self.n_outputs_ = y.shape[1]
        check_consistent_length(X, y, sample_weight)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
        if self.strategy == 'mean':
            self.constant_ = np.average(y, axis=0, weights=sample_weight)
        elif self.strategy == 'median':
            if sample_weight is None:
                self.constant_ = np.median(y, axis=0)
            else:
                self.constant_ = [_weighted_percentile(y[:, k], sample_weight, percentile=50.0) for k in range(self.n_outputs_)]
        elif self.strategy == 'quantile':
            if self.quantile is None:
                raise ValueError("When using `strategy='quantile', you have to specify the desired quantile in the range [0, 1].")
            percentile = self.quantile * 100.0
            if sample_weight is None:
                self.constant_ = np.percentile(y, axis=0, q=percentile)
            else:
                self.constant_ = [_weighted_percentile(y[:, k], sample_weight, percentile=percentile) for k in range(self.n_outputs_)]
        elif self.strategy == 'constant':
            if self.constant is None:
                raise TypeError('Constant target value has to be specified when the constant strategy is used.')
            self.constant_ = check_array(self.constant, accept_sparse=['csr', 'csc', 'coo'], ensure_2d=False, ensure_min_samples=0)
            if self.n_outputs_ != 1 and self.constant_.shape[0] != y.shape[1]:
                raise ValueError('Constant target value should have shape (%d, 1).' % y.shape[1])
        self.constant_ = np.reshape(self.constant_, (1, -1))
        return self

    def predict(self, X, return_std=False):
        """Perform classification on test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        return_std : bool, default=False
            Whether to return the standard deviation of posterior prediction.
            All zeros in this case.

            .. versionadded:: 0.20

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted target values for X.

        y_std : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Standard deviation of predictive distribution of query points.
        """
        check_is_fitted(self)
        n_samples = _num_samples(X)
        y = np.full((n_samples, self.n_outputs_), self.constant_, dtype=np.array(self.constant_).dtype)
        y_std = np.zeros((n_samples, self.n_outputs_))
        if self.n_outputs_ == 1:
            y = np.ravel(y)
            y_std = np.ravel(y_std)
        return (y, y_std) if return_std else y

    def _more_tags(self):
        return {'poor_score': True, 'no_validation': True}

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as `(1 - u/v)`, where `u` is the
        residual sum of squares `((y_true - y_pred) ** 2).sum()` and `v` is the
        total sum of squares `((y_true - y_true.mean()) ** 2).sum()`. The best
        possible score is 1.0 and it can be negative (because the model can be
        arbitrarily worse). A constant model that always predicts the expected
        value of y, disregarding the input features, would get a R^2 score of
        0.0.

        Parameters
        ----------
        X : None or array-like of shape (n_samples, n_features)
            Test samples. Passing None as test samples gives the same result
            as passing real test samples, since `DummyRegressor`
            operates independently of the sampled observations.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            R^2 of `self.predict(X)` w.r.t. y.
        """
        if X is None:
            X = np.zeros(shape=(len(y), 1))
        return super().score(X, y, sample_weight)