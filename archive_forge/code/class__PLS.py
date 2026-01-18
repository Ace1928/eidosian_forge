import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy.linalg import svd
from ..base import (
from ..exceptions import ConvergenceWarning
from ..utils import check_array, check_consistent_length
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import svd_flip
from ..utils.fixes import parse_version, sp_version
from ..utils.validation import FLOAT_DTYPES, check_is_fitted
class _PLS(ClassNamePrefixFeaturesOutMixin, TransformerMixin, RegressorMixin, MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    """Partial Least Squares (PLS)

    This class implements the generic PLS algorithm.

    Main ref: Wegelin, a survey of Partial Least Squares (PLS) methods,
    with emphasis on the two-block case
    https://stat.uw.edu/sites/default/files/files/reports/2000/tr371.pdf
    """
    _parameter_constraints: dict = {'n_components': [Interval(Integral, 1, None, closed='left')], 'scale': ['boolean'], 'deflation_mode': [StrOptions({'regression', 'canonical'})], 'mode': [StrOptions({'A', 'B'})], 'algorithm': [StrOptions({'svd', 'nipals'})], 'max_iter': [Interval(Integral, 1, None, closed='left')], 'tol': [Interval(Real, 0, None, closed='left')], 'copy': ['boolean']}

    @abstractmethod
    def __init__(self, n_components=2, *, scale=True, deflation_mode='regression', mode='A', algorithm='nipals', max_iter=500, tol=1e-06, copy=True):
        self.n_components = n_components
        self.deflation_mode = deflation_mode
        self.mode = mode
        self.scale = scale
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, Y):
        """Fit model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.

        Returns
        -------
        self : object
            Fitted model.
        """
        check_consistent_length(X, Y)
        X = self._validate_data(X, dtype=np.float64, copy=self.copy, ensure_min_samples=2)
        Y = check_array(Y, input_name='Y', dtype=np.float64, copy=self.copy, ensure_2d=False)
        if Y.ndim == 1:
            self._predict_1d = True
            Y = Y.reshape(-1, 1)
        else:
            self._predict_1d = False
        n = X.shape[0]
        p = X.shape[1]
        q = Y.shape[1]
        n_components = self.n_components
        rank_upper_bound = p if self.deflation_mode == 'regression' else min(n, p, q)
        if n_components > rank_upper_bound:
            raise ValueError(f'`n_components` upper bound is {rank_upper_bound}. Got {n_components} instead. Reduce `n_components`.')
        self._norm_y_weights = self.deflation_mode == 'canonical'
        norm_y_weights = self._norm_y_weights
        Xk, Yk, self._x_mean, self._y_mean, self._x_std, self._y_std = _center_scale_xy(X, Y, self.scale)
        self.x_weights_ = np.zeros((p, n_components))
        self.y_weights_ = np.zeros((q, n_components))
        self._x_scores = np.zeros((n, n_components))
        self._y_scores = np.zeros((n, n_components))
        self.x_loadings_ = np.zeros((p, n_components))
        self.y_loadings_ = np.zeros((q, n_components))
        self.n_iter_ = []
        Y_eps = np.finfo(Yk.dtype).eps
        for k in range(n_components):
            if self.algorithm == 'nipals':
                Yk_mask = np.all(np.abs(Yk) < 10 * Y_eps, axis=0)
                Yk[:, Yk_mask] = 0.0
                try:
                    x_weights, y_weights, n_iter_ = _get_first_singular_vectors_power_method(Xk, Yk, mode=self.mode, max_iter=self.max_iter, tol=self.tol, norm_y_weights=norm_y_weights)
                except StopIteration as e:
                    if str(e) != 'Y residual is constant':
                        raise
                    warnings.warn(f'Y residual is constant at iteration {k}')
                    break
                self.n_iter_.append(n_iter_)
            elif self.algorithm == 'svd':
                x_weights, y_weights = _get_first_singular_vectors_svd(Xk, Yk)
            _svd_flip_1d(x_weights, y_weights)
            x_scores = np.dot(Xk, x_weights)
            if norm_y_weights:
                y_ss = 1
            else:
                y_ss = np.dot(y_weights, y_weights)
            y_scores = np.dot(Yk, y_weights) / y_ss
            x_loadings = np.dot(x_scores, Xk) / np.dot(x_scores, x_scores)
            Xk -= np.outer(x_scores, x_loadings)
            if self.deflation_mode == 'canonical':
                y_loadings = np.dot(y_scores, Yk) / np.dot(y_scores, y_scores)
                Yk -= np.outer(y_scores, y_loadings)
            if self.deflation_mode == 'regression':
                y_loadings = np.dot(x_scores, Yk) / np.dot(x_scores, x_scores)
                Yk -= np.outer(x_scores, y_loadings)
            self.x_weights_[:, k] = x_weights
            self.y_weights_[:, k] = y_weights
            self._x_scores[:, k] = x_scores
            self._y_scores[:, k] = y_scores
            self.x_loadings_[:, k] = x_loadings
            self.y_loadings_[:, k] = y_loadings
        self.x_rotations_ = np.dot(self.x_weights_, pinv2(np.dot(self.x_loadings_.T, self.x_weights_), check_finite=False))
        self.y_rotations_ = np.dot(self.y_weights_, pinv2(np.dot(self.y_loadings_.T, self.y_weights_), check_finite=False))
        self.coef_ = np.dot(self.x_rotations_, self.y_loadings_.T)
        self.coef_ = (self.coef_ * self._y_std).T
        self.intercept_ = self._y_mean
        self._n_features_out = self.x_rotations_.shape[1]
        return self

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Y : array-like of shape (n_samples, n_targets), default=None
            Target vectors.

        copy : bool, default=True
            Whether to copy `X` and `Y`, or perform in-place normalization.

        Returns
        -------
        x_scores, y_scores : array-like or tuple of array-like
            Return `x_scores` if `Y` is not given, `(x_scores, y_scores)` otherwise.
        """
        check_is_fitted(self)
        X = self._validate_data(X, copy=copy, dtype=FLOAT_DTYPES, reset=False)
        X -= self._x_mean
        X /= self._x_std
        x_scores = np.dot(X, self.x_rotations_)
        if Y is not None:
            Y = check_array(Y, input_name='Y', ensure_2d=False, copy=copy, dtype=FLOAT_DTYPES)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            Y -= self._y_mean
            Y /= self._y_std
            y_scores = np.dot(Y, self.y_rotations_)
            return (x_scores, y_scores)
        return x_scores

    def inverse_transform(self, X, Y=None):
        """Transform data back to its original space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            New data, where `n_samples` is the number of samples
            and `n_components` is the number of pls components.

        Y : array-like of shape (n_samples, n_components)
            New target, where `n_samples` is the number of samples
            and `n_components` is the number of pls components.

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_features)
            Return the reconstructed `X` data.

        Y_reconstructed : ndarray of shape (n_samples, n_targets)
            Return the reconstructed `X` target. Only returned when `Y` is given.

        Notes
        -----
        This transformation will only be exact if `n_components=n_features`.
        """
        check_is_fitted(self)
        X = check_array(X, input_name='X', dtype=FLOAT_DTYPES)
        X_reconstructed = np.matmul(X, self.x_loadings_.T)
        X_reconstructed *= self._x_std
        X_reconstructed += self._x_mean
        if Y is not None:
            Y = check_array(Y, input_name='Y', dtype=FLOAT_DTYPES)
            Y_reconstructed = np.matmul(Y, self.y_loadings_.T)
            Y_reconstructed *= self._y_std
            Y_reconstructed += self._y_mean
            return (X_reconstructed, Y_reconstructed)
        return X_reconstructed

    def predict(self, X, copy=True):
        """Predict targets of given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        copy : bool, default=True
            Whether to copy `X` and `Y`, or perform in-place normalization.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.

        Notes
        -----
        This call requires the estimation of a matrix of shape
        `(n_features, n_targets)`, which may be an issue in high dimensional
        space.
        """
        check_is_fitted(self)
        X = self._validate_data(X, copy=copy, dtype=FLOAT_DTYPES, reset=False)
        X -= self._x_mean
        X /= self._x_std
        Ypred = X @ self.coef_.T + self.intercept_
        return Ypred.ravel() if self._predict_1d else Ypred

    def fit_transform(self, X, y=None):
        """Learn and apply the dimension reduction on the train data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples, n_targets), default=None
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.

        Returns
        -------
        self : ndarray of shape (n_samples, n_components)
            Return `x_scores` if `Y` is not given, `(x_scores, y_scores)` otherwise.
        """
        return self.fit(X, y).transform(X, y)

    def _more_tags(self):
        return {'poor_score': True, 'requires_y': False}