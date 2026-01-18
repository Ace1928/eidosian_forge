import numbers
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Integral, Real
import numpy as np
from scipy import linalg, optimize, sparse
from scipy.sparse import linalg as sp_linalg
from ..base import MultiOutputMixin, RegressorMixin, _fit_context, is_classifier
from ..exceptions import ConvergenceWarning
from ..metrics import check_scoring, get_scorer_names
from ..model_selection import GridSearchCV
from ..preprocessing import LabelBinarizer
from ..utils import (
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.fixes import _sparse_linalg_cg
from ..utils.metadata_routing import (
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._base import LinearClassifierMixin, LinearModel, _preprocess_data, _rescale_data
from ._sag import sag_solver
class _RidgeGCV(LinearModel):
    """Ridge regression with built-in Leave-one-out Cross-Validation.

    This class is not intended to be used directly. Use RidgeCV instead.

    Notes
    -----

    We want to solve (K + alpha*Id)c = y,
    where K = X X^T is the kernel matrix.

    Let G = (K + alpha*Id).

    Dual solution: c = G^-1y
    Primal solution: w = X^T c

    Compute eigendecomposition K = Q V Q^T.
    Then G^-1 = Q (V + alpha*Id)^-1 Q^T,
    where (V + alpha*Id) is diagonal.
    It is thus inexpensive to inverse for many alphas.

    Let loov be the vector of prediction values for each example
    when the model was fitted with all examples but this example.

    loov = (KG^-1Y - diag(KG^-1)Y) / diag(I-KG^-1)

    Let looe be the vector of prediction errors for each example
    when the model was fitted with all examples but this example.

    looe = y - loov = c / diag(G^-1)

    The best score (negative mean squared error or user-provided scoring) is
    stored in the `best_score_` attribute, and the selected hyperparameter in
    `alpha_`.

    References
    ----------
    http://cbcl.mit.edu/publications/ps/MIT-CSAIL-TR-2007-025.pdf
    https://www.mit.edu/~9.520/spring07/Classes/rlsslides.pdf
    """

    def __init__(self, alphas=(0.1, 1.0, 10.0), *, fit_intercept=True, scoring=None, copy_X=True, gcv_mode=None, store_cv_values=False, is_clf=False, alpha_per_target=False):
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.scoring = scoring
        self.copy_X = copy_X
        self.gcv_mode = gcv_mode
        self.store_cv_values = store_cv_values
        self.is_clf = is_clf
        self.alpha_per_target = alpha_per_target

    @staticmethod
    def _decomp_diag(v_prime, Q):
        return (v_prime * Q ** 2).sum(axis=-1)

    @staticmethod
    def _diag_dot(D, B):
        if len(B.shape) > 1:
            D = D[(slice(None),) + (np.newaxis,) * (len(B.shape) - 1)]
        return D * B

    def _compute_gram(self, X, sqrt_sw):
        """Computes the Gram matrix XX^T with possible centering.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The preprocessed design matrix.

        sqrt_sw : ndarray of shape (n_samples,)
            square roots of sample weights

        Returns
        -------
        gram : ndarray of shape (n_samples, n_samples)
            The Gram matrix.
        X_mean : ndarray of shape (n_feature,)
            The weighted mean of ``X`` for each feature.

        Notes
        -----
        When X is dense the centering has been done in preprocessing
        so the mean is 0 and we just compute XX^T.

        When X is sparse it has not been centered in preprocessing, but it has
        been scaled by sqrt(sample weights).

        When self.fit_intercept is False no centering is done.

        The centered X is never actually computed because centering would break
        the sparsity of X.
        """
        center = self.fit_intercept and sparse.issparse(X)
        if not center:
            X_mean = np.zeros(X.shape[1], dtype=X.dtype)
            return (safe_sparse_dot(X, X.T, dense_output=True), X_mean)
        n_samples = X.shape[0]
        sample_weight_matrix = sparse.dia_matrix((sqrt_sw, 0), shape=(n_samples, n_samples))
        X_weighted = sample_weight_matrix.dot(X)
        X_mean, _ = mean_variance_axis(X_weighted, axis=0)
        X_mean *= n_samples / sqrt_sw.dot(sqrt_sw)
        X_mX = sqrt_sw[:, None] * safe_sparse_dot(X_mean, X.T, dense_output=True)
        X_mX_m = np.outer(sqrt_sw, sqrt_sw) * np.dot(X_mean, X_mean)
        return (safe_sparse_dot(X, X.T, dense_output=True) + X_mX_m - X_mX - X_mX.T, X_mean)

    def _compute_covariance(self, X, sqrt_sw):
        """Computes covariance matrix X^TX with possible centering.

        Parameters
        ----------
        X : sparse matrix of shape (n_samples, n_features)
            The preprocessed design matrix.

        sqrt_sw : ndarray of shape (n_samples,)
            square roots of sample weights

        Returns
        -------
        covariance : ndarray of shape (n_features, n_features)
            The covariance matrix.
        X_mean : ndarray of shape (n_feature,)
            The weighted mean of ``X`` for each feature.

        Notes
        -----
        Since X is sparse it has not been centered in preprocessing, but it has
        been scaled by sqrt(sample weights).

        When self.fit_intercept is False no centering is done.

        The centered X is never actually computed because centering would break
        the sparsity of X.
        """
        if not self.fit_intercept:
            X_mean = np.zeros(X.shape[1], dtype=X.dtype)
            return (safe_sparse_dot(X.T, X, dense_output=True), X_mean)
        n_samples = X.shape[0]
        sample_weight_matrix = sparse.dia_matrix((sqrt_sw, 0), shape=(n_samples, n_samples))
        X_weighted = sample_weight_matrix.dot(X)
        X_mean, _ = mean_variance_axis(X_weighted, axis=0)
        X_mean = X_mean * n_samples / sqrt_sw.dot(sqrt_sw)
        weight_sum = sqrt_sw.dot(sqrt_sw)
        return (safe_sparse_dot(X.T, X, dense_output=True) - weight_sum * np.outer(X_mean, X_mean), X_mean)

    def _sparse_multidot_diag(self, X, A, X_mean, sqrt_sw):
        """Compute the diagonal of (X - X_mean).dot(A).dot((X - X_mean).T)
        without explicitly centering X nor computing X.dot(A)
        when X is sparse.

        Parameters
        ----------
        X : sparse matrix of shape (n_samples, n_features)

        A : ndarray of shape (n_features, n_features)

        X_mean : ndarray of shape (n_features,)

        sqrt_sw : ndarray of shape (n_features,)
            square roots of sample weights

        Returns
        -------
        diag : np.ndarray, shape (n_samples,)
            The computed diagonal.
        """
        intercept_col = scale = sqrt_sw
        batch_size = X.shape[1]
        diag = np.empty(X.shape[0], dtype=X.dtype)
        for start in range(0, X.shape[0], batch_size):
            batch = slice(start, min(X.shape[0], start + batch_size), 1)
            X_batch = np.empty((X[batch].shape[0], X.shape[1] + self.fit_intercept), dtype=X.dtype)
            if self.fit_intercept:
                X_batch[:, :-1] = X[batch].toarray() - X_mean * scale[batch][:, None]
                X_batch[:, -1] = intercept_col[batch]
            else:
                X_batch = X[batch].toarray()
            diag[batch] = (X_batch.dot(A) * X_batch).sum(axis=1)
        return diag

    def _eigen_decompose_gram(self, X, y, sqrt_sw):
        """Eigendecomposition of X.X^T, used when n_samples <= n_features."""
        K, X_mean = self._compute_gram(X, sqrt_sw)
        if self.fit_intercept:
            K += np.outer(sqrt_sw, sqrt_sw)
        eigvals, Q = linalg.eigh(K)
        QT_y = np.dot(Q.T, y)
        return (X_mean, eigvals, Q, QT_y)

    def _solve_eigen_gram(self, alpha, y, sqrt_sw, X_mean, eigvals, Q, QT_y):
        """Compute dual coefficients and diagonal of G^-1.

        Used when we have a decomposition of X.X^T (n_samples <= n_features).
        """
        w = 1.0 / (eigvals + alpha)
        if self.fit_intercept:
            normalized_sw = sqrt_sw / np.linalg.norm(sqrt_sw)
            intercept_dim = _find_smallest_angle(normalized_sw, Q)
            w[intercept_dim] = 0
        c = np.dot(Q, self._diag_dot(w, QT_y))
        G_inverse_diag = self._decomp_diag(w, Q)
        if len(y.shape) != 1:
            G_inverse_diag = G_inverse_diag[:, np.newaxis]
        return (G_inverse_diag, c)

    def _eigen_decompose_covariance(self, X, y, sqrt_sw):
        """Eigendecomposition of X^T.X, used when n_samples > n_features
        and X is sparse.
        """
        n_samples, n_features = X.shape
        cov = np.empty((n_features + 1, n_features + 1), dtype=X.dtype)
        cov[:-1, :-1], X_mean = self._compute_covariance(X, sqrt_sw)
        if not self.fit_intercept:
            cov = cov[:-1, :-1]
        else:
            cov[-1] = 0
            cov[:, -1] = 0
            cov[-1, -1] = sqrt_sw.dot(sqrt_sw)
        nullspace_dim = max(0, n_features - n_samples)
        eigvals, V = linalg.eigh(cov)
        eigvals = eigvals[nullspace_dim:]
        V = V[:, nullspace_dim:]
        return (X_mean, eigvals, V, X)

    def _solve_eigen_covariance_no_intercept(self, alpha, y, sqrt_sw, X_mean, eigvals, V, X):
        """Compute dual coefficients and diagonal of G^-1.

        Used when we have a decomposition of X^T.X
        (n_samples > n_features and X is sparse), and not fitting an intercept.
        """
        w = 1 / (eigvals + alpha)
        A = (V * w).dot(V.T)
        AXy = A.dot(safe_sparse_dot(X.T, y, dense_output=True))
        y_hat = safe_sparse_dot(X, AXy, dense_output=True)
        hat_diag = self._sparse_multidot_diag(X, A, X_mean, sqrt_sw)
        if len(y.shape) != 1:
            hat_diag = hat_diag[:, np.newaxis]
        return ((1 - hat_diag) / alpha, (y - y_hat) / alpha)

    def _solve_eigen_covariance_intercept(self, alpha, y, sqrt_sw, X_mean, eigvals, V, X):
        """Compute dual coefficients and diagonal of G^-1.

        Used when we have a decomposition of X^T.X
        (n_samples > n_features and X is sparse),
        and we are fitting an intercept.
        """
        intercept_sv = np.zeros(V.shape[0])
        intercept_sv[-1] = 1
        intercept_dim = _find_smallest_angle(intercept_sv, V)
        w = 1 / (eigvals + alpha)
        w[intercept_dim] = 1 / eigvals[intercept_dim]
        A = (V * w).dot(V.T)
        X_op = _X_CenterStackOp(X, X_mean, sqrt_sw)
        AXy = A.dot(X_op.T.dot(y))
        y_hat = X_op.dot(AXy)
        hat_diag = self._sparse_multidot_diag(X, A, X_mean, sqrt_sw)
        if len(y.shape) != 1:
            hat_diag = hat_diag[:, np.newaxis]
        return ((1 - hat_diag) / alpha, (y - y_hat) / alpha)

    def _solve_eigen_covariance(self, alpha, y, sqrt_sw, X_mean, eigvals, V, X):
        """Compute dual coefficients and diagonal of G^-1.

        Used when we have a decomposition of X^T.X
        (n_samples > n_features and X is sparse).
        """
        if self.fit_intercept:
            return self._solve_eigen_covariance_intercept(alpha, y, sqrt_sw, X_mean, eigvals, V, X)
        return self._solve_eigen_covariance_no_intercept(alpha, y, sqrt_sw, X_mean, eigvals, V, X)

    def _svd_decompose_design_matrix(self, X, y, sqrt_sw):
        X_mean = np.zeros(X.shape[1], dtype=X.dtype)
        if self.fit_intercept:
            intercept_column = sqrt_sw[:, None]
            X = np.hstack((X, intercept_column))
        U, singvals, _ = linalg.svd(X, full_matrices=0)
        singvals_sq = singvals ** 2
        UT_y = np.dot(U.T, y)
        return (X_mean, singvals_sq, U, UT_y)

    def _solve_svd_design_matrix(self, alpha, y, sqrt_sw, X_mean, singvals_sq, U, UT_y):
        """Compute dual coefficients and diagonal of G^-1.

        Used when we have an SVD decomposition of X
        (n_samples > n_features and X is dense).
        """
        w = (singvals_sq + alpha) ** (-1) - alpha ** (-1)
        if self.fit_intercept:
            normalized_sw = sqrt_sw / np.linalg.norm(sqrt_sw)
            intercept_dim = _find_smallest_angle(normalized_sw, U)
            w[intercept_dim] = -alpha ** (-1)
        c = np.dot(U, self._diag_dot(w, UT_y)) + alpha ** (-1) * y
        G_inverse_diag = self._decomp_diag(w, U) + alpha ** (-1)
        if len(y.shape) != 1:
            G_inverse_diag = G_inverse_diag[:, np.newaxis]
        return (G_inverse_diag, c)

    def fit(self, X, y, sample_weight=None):
        """Fit Ridge regression model with gcv.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Training data. Will be cast to float64 if necessary.

        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to float64 if necessary.

        sample_weight : float or ndarray of shape (n_samples,), default=None
            Individual weights for each sample. If given a float, every sample
            will have the same weight. Note that the scale of `sample_weight`
            has an impact on the loss; i.e. multiplying all weights by `k`
            is equivalent to setting `alpha / k`.

        Returns
        -------
        self : object
        """
        X, y = self._validate_data(X, y, accept_sparse=['csr', 'csc', 'coo'], dtype=[np.float64], multi_output=True, y_numeric=True)
        assert not (self.is_clf and self.alpha_per_target)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self.alphas = np.asarray(self.alphas)
        X, y, X_offset, y_offset, X_scale = _preprocess_data(X, y, fit_intercept=self.fit_intercept, copy=self.copy_X, sample_weight=sample_weight)
        gcv_mode = _check_gcv_mode(X, self.gcv_mode)
        if gcv_mode == 'eigen':
            decompose = self._eigen_decompose_gram
            solve = self._solve_eigen_gram
        elif gcv_mode == 'svd':
            if sparse.issparse(X):
                decompose = self._eigen_decompose_covariance
                solve = self._solve_eigen_covariance
            else:
                decompose = self._svd_decompose_design_matrix
                solve = self._solve_svd_design_matrix
        n_samples = X.shape[0]
        if sample_weight is not None:
            X, y, sqrt_sw = _rescale_data(X, y, sample_weight)
        else:
            sqrt_sw = np.ones(n_samples, dtype=X.dtype)
        X_mean, *decomposition = decompose(X, y, sqrt_sw)
        scorer = check_scoring(self, scoring=self.scoring, allow_none=True)
        error = scorer is None
        n_y = 1 if len(y.shape) == 1 else y.shape[1]
        n_alphas = 1 if np.ndim(self.alphas) == 0 else len(self.alphas)
        if self.store_cv_values:
            self.cv_values_ = np.empty((n_samples * n_y, n_alphas), dtype=X.dtype)
        best_coef, best_score, best_alpha = (None, None, None)
        for i, alpha in enumerate(np.atleast_1d(self.alphas)):
            G_inverse_diag, c = solve(float(alpha), y, sqrt_sw, X_mean, *decomposition)
            if error:
                squared_errors = (c / G_inverse_diag) ** 2
                if self.alpha_per_target:
                    alpha_score = -squared_errors.mean(axis=0)
                else:
                    alpha_score = -squared_errors.mean()
                if self.store_cv_values:
                    self.cv_values_[:, i] = squared_errors.ravel()
            else:
                predictions = y - c / G_inverse_diag
                if self.store_cv_values:
                    self.cv_values_[:, i] = predictions.ravel()
                if self.is_clf:
                    identity_estimator = _IdentityClassifier(classes=np.arange(n_y))
                    alpha_score = scorer(identity_estimator, predictions, y.argmax(axis=1))
                else:
                    identity_estimator = _IdentityRegressor()
                    if self.alpha_per_target:
                        alpha_score = np.array([scorer(identity_estimator, predictions[:, j], y[:, j]) for j in range(n_y)])
                    else:
                        alpha_score = scorer(identity_estimator, predictions.ravel(), y.ravel())
            if best_score is None:
                if self.alpha_per_target and n_y > 1:
                    best_coef = c
                    best_score = np.atleast_1d(alpha_score)
                    best_alpha = np.full(n_y, alpha)
                else:
                    best_coef = c
                    best_score = alpha_score
                    best_alpha = alpha
            elif self.alpha_per_target and n_y > 1:
                to_update = alpha_score > best_score
                best_coef[:, to_update] = c[:, to_update]
                best_score[to_update] = alpha_score[to_update]
                best_alpha[to_update] = alpha
            elif alpha_score > best_score:
                best_coef, best_score, best_alpha = (c, alpha_score, alpha)
        self.alpha_ = best_alpha
        self.best_score_ = best_score
        self.dual_coef_ = best_coef
        self.coef_ = safe_sparse_dot(self.dual_coef_.T, X)
        if sparse.issparse(X):
            X_offset = X_mean * X_scale
        else:
            X_offset += X_mean * X_scale
        self._set_intercept(X_offset, y_offset, X_scale)
        if self.store_cv_values:
            if len(y.shape) == 1:
                cv_values_shape = (n_samples, n_alphas)
            else:
                cv_values_shape = (n_samples, n_y, n_alphas)
            self.cv_values_ = self.cv_values_.reshape(cv_values_shape)
        return self