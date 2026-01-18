import math
import warnings
from numbers import Real
import numpy as np
from scipy import interpolate
from scipy.stats import spearmanr
from ._isotonic import _inplace_contiguous_isotonic_regression, _make_unique
from .base import BaseEstimator, RegressorMixin, TransformerMixin, _fit_context
from .utils import check_array, check_consistent_length
from .utils._param_validation import Interval, StrOptions, validate_params
from .utils.validation import _check_sample_weight, check_is_fitted
class IsotonicRegression(RegressorMixin, TransformerMixin, BaseEstimator):
    """Isotonic regression model.

    Read more in the :ref:`User Guide <isotonic>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    y_min : float, default=None
        Lower bound on the lowest predicted value (the minimum value may
        still be higher). If not set, defaults to -inf.

    y_max : float, default=None
        Upper bound on the highest predicted value (the maximum may still be
        lower). If not set, defaults to +inf.

    increasing : bool or 'auto', default=True
        Determines whether the predictions should be constrained to increase
        or decrease with `X`. 'auto' will decide based on the Spearman
        correlation estimate's sign.

    out_of_bounds : {'nan', 'clip', 'raise'}, default='nan'
        Handles how `X` values outside of the training domain are handled
        during prediction.

        - 'nan', predictions will be NaN.
        - 'clip', predictions will be set to the value corresponding to
          the nearest train interval endpoint.
        - 'raise', a `ValueError` is raised.

    Attributes
    ----------
    X_min_ : float
        Minimum value of input array `X_` for left bound.

    X_max_ : float
        Maximum value of input array `X_` for right bound.

    X_thresholds_ : ndarray of shape (n_thresholds,)
        Unique ascending `X` values used to interpolate
        the y = f(X) monotonic function.

        .. versionadded:: 0.24

    y_thresholds_ : ndarray of shape (n_thresholds,)
        De-duplicated `y` values suitable to interpolate the y = f(X)
        monotonic function.

        .. versionadded:: 0.24

    f_ : function
        The stepwise interpolating function that covers the input domain ``X``.

    increasing_ : bool
        Inferred value for ``increasing``.

    See Also
    --------
    sklearn.linear_model.LinearRegression : Ordinary least squares Linear
        Regression.
    sklearn.ensemble.HistGradientBoostingRegressor : Gradient boosting that
        is a non-parametric model accepting monotonicity constraints.
    isotonic_regression : Function to solve the isotonic regression model.

    Notes
    -----
    Ties are broken using the secondary method from de Leeuw, 1977.

    References
    ----------
    Isotonic Median Regression: A Linear Programming Approach
    Nilotpal Chakravarti
    Mathematics of Operations Research
    Vol. 14, No. 2 (May, 1989), pp. 303-308

    Isotone Optimization in R : Pool-Adjacent-Violators
    Algorithm (PAVA) and Active Set Methods
    de Leeuw, Hornik, Mair
    Journal of Statistical Software 2009

    Correctness of Kruskal's algorithms for monotone regression with ties
    de Leeuw, Psychometrica, 1977

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.isotonic import IsotonicRegression
    >>> X, y = make_regression(n_samples=10, n_features=1, random_state=41)
    >>> iso_reg = IsotonicRegression().fit(X, y)
    >>> iso_reg.predict([.1, .2])
    array([1.8628..., 3.7256...])
    """
    _parameter_constraints: dict = {'y_min': [Interval(Real, None, None, closed='both'), None], 'y_max': [Interval(Real, None, None, closed='both'), None], 'increasing': ['boolean', StrOptions({'auto'})], 'out_of_bounds': [StrOptions({'nan', 'clip', 'raise'})]}

    def __init__(self, *, y_min=None, y_max=None, increasing=True, out_of_bounds='nan'):
        self.y_min = y_min
        self.y_max = y_max
        self.increasing = increasing
        self.out_of_bounds = out_of_bounds

    def _check_input_data_shape(self, X):
        if not (X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1)):
            msg = 'Isotonic regression input X should be a 1d array or 2d array with 1 feature'
            raise ValueError(msg)

    def _build_f(self, X, y):
        """Build the f_ interp1d function."""
        bounds_error = self.out_of_bounds == 'raise'
        if len(y) == 1:
            self.f_ = lambda x: y.repeat(x.shape)
        else:
            self.f_ = interpolate.interp1d(X, y, kind='linear', bounds_error=bounds_error)

    def _build_y(self, X, y, sample_weight, trim_duplicates=True):
        """Build the y_ IsotonicRegression."""
        self._check_input_data_shape(X)
        X = X.reshape(-1)
        if self.increasing == 'auto':
            self.increasing_ = check_increasing(X, y)
        else:
            self.increasing_ = self.increasing
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        mask = sample_weight > 0
        X, y, sample_weight = (X[mask], y[mask], sample_weight[mask])
        order = np.lexsort((y, X))
        X, y, sample_weight = [array[order] for array in [X, y, sample_weight]]
        unique_X, unique_y, unique_sample_weight = _make_unique(X, y, sample_weight)
        X = unique_X
        y = isotonic_regression(unique_y, sample_weight=unique_sample_weight, y_min=self.y_min, y_max=self.y_max, increasing=self.increasing_)
        self.X_min_, self.X_max_ = (np.min(X), np.max(X))
        if trim_duplicates:
            keep_data = np.ones((len(y),), dtype=bool)
            keep_data[1:-1] = np.logical_or(np.not_equal(y[1:-1], y[:-2]), np.not_equal(y[1:-1], y[2:]))
            return (X[keep_data], y[keep_data])
        else:
            return (X, y)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Training data.

            .. versionchanged:: 0.24
               Also accepts 2d array with 1 feature.

        y : array-like of shape (n_samples,)
            Training target.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights. If set to None, all weights will be set to 1 (equal
            weights).

        Returns
        -------
        self : object
            Returns an instance of self.

        Notes
        -----
        X is stored for future use, as :meth:`transform` needs X to interpolate
        new input data.
        """
        check_params = dict(accept_sparse=False, ensure_2d=False)
        X = check_array(X, input_name='X', dtype=[np.float64, np.float32], **check_params)
        y = check_array(y, input_name='y', dtype=X.dtype, **check_params)
        check_consistent_length(X, y, sample_weight)
        X, y = self._build_y(X, y, sample_weight)
        self.X_thresholds_, self.y_thresholds_ = (X, y)
        self._build_f(X, y)
        return self

    def _transform(self, T):
        """`_transform` is called by both `transform` and `predict` methods.

        Since `transform` is wrapped to output arrays of specific types (e.g.
        NumPy arrays, pandas DataFrame), we cannot make `predict` call `transform`
        directly.

        The above behaviour could be changed in the future, if we decide to output
        other type of arrays when calling `predict`.
        """
        if hasattr(self, 'X_thresholds_'):
            dtype = self.X_thresholds_.dtype
        else:
            dtype = np.float64
        T = check_array(T, dtype=dtype, ensure_2d=False)
        self._check_input_data_shape(T)
        T = T.reshape(-1)
        if self.out_of_bounds == 'clip':
            T = np.clip(T, self.X_min_, self.X_max_)
        res = self.f_(T)
        res = res.astype(T.dtype)
        return res

    def transform(self, T):
        """Transform new data by linear interpolation.

        Parameters
        ----------
        T : array-like of shape (n_samples,) or (n_samples, 1)
            Data to transform.

            .. versionchanged:: 0.24
               Also accepts 2d array with 1 feature.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The transformed data.
        """
        return self._transform(T)

    def predict(self, T):
        """Predict new data by linear interpolation.

        Parameters
        ----------
        T : array-like of shape (n_samples,) or (n_samples, 1)
            Data to transform.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Transformed data.
        """
        return self._transform(T)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Ignored.

        Returns
        -------
        feature_names_out : ndarray of str objects
            An ndarray with one string i.e. ["isotonicregression0"].
        """
        check_is_fitted(self, 'f_')
        class_name = self.__class__.__name__.lower()
        return np.asarray([f'{class_name}0'], dtype=object)

    def __getstate__(self):
        """Pickle-protocol - return state of the estimator."""
        state = super().__getstate__()
        state.pop('f_', None)
        return state

    def __setstate__(self, state):
        """Pickle-protocol - set state of the estimator.

        We need to rebuild the interpolation function.
        """
        super().__setstate__(state)
        if hasattr(self, 'X_thresholds_') and hasattr(self, 'y_thresholds_'):
            self._build_f(self.X_thresholds_, self.y_thresholds_)

    def _more_tags(self):
        return {'X_types': ['1darray']}