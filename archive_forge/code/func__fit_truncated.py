from math import log, sqrt
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from scipy.sparse import issparse
from scipy.sparse.linalg import svds
from scipy.special import gammaln
from ..base import _fit_context
from ..utils import check_random_state
from ..utils._arpack import _init_arpack_v0
from ..utils._array_api import _convert_to_numpy, get_namespace
from ..utils._param_validation import Interval, RealNotInt, StrOptions
from ..utils.extmath import fast_logdet, randomized_svd, stable_cumsum, svd_flip
from ..utils.sparsefuncs import _implicit_column_offset, mean_variance_axis
from ..utils.validation import check_is_fitted
from ._base import _BasePCA
def _fit_truncated(self, X, n_components, svd_solver):
    """Fit the model by computing truncated SVD (by ARPACK or randomized)
        on X.
        """
    xp, _ = get_namespace(X)
    n_samples, n_features = X.shape
    if isinstance(n_components, str):
        raise ValueError("n_components=%r cannot be a string with svd_solver='%s'" % (n_components, svd_solver))
    elif not 1 <= n_components <= min(n_samples, n_features):
        raise ValueError("n_components=%r must be between 1 and min(n_samples, n_features)=%r with svd_solver='%s'" % (n_components, min(n_samples, n_features), svd_solver))
    elif svd_solver == 'arpack' and n_components == min(n_samples, n_features):
        raise ValueError("n_components=%r must be strictly less than min(n_samples, n_features)=%r with svd_solver='%s'" % (n_components, min(n_samples, n_features), svd_solver))
    random_state = check_random_state(self.random_state)
    total_var = None
    if issparse(X):
        self.mean_, var = mean_variance_axis(X, axis=0)
        total_var = var.sum() * n_samples / (n_samples - 1)
        X = _implicit_column_offset(X, self.mean_)
    else:
        self.mean_ = xp.mean(X, axis=0)
        X -= self.mean_
    if svd_solver == 'arpack':
        v0 = _init_arpack_v0(min(X.shape), random_state)
        U, S, Vt = svds(X, k=n_components, tol=self.tol, v0=v0)
        S = S[::-1]
        U, Vt = svd_flip(U[:, ::-1], Vt[::-1])
    elif svd_solver == 'randomized':
        U, S, Vt = randomized_svd(X, n_components=n_components, n_oversamples=self.n_oversamples, n_iter=self.iterated_power, power_iteration_normalizer=self.power_iteration_normalizer, flip_sign=True, random_state=random_state)
    self.n_samples_ = n_samples
    self.components_ = Vt
    self.n_components_ = n_components
    self.explained_variance_ = S ** 2 / (n_samples - 1)
    if total_var is None:
        N = X.shape[0] - 1
        X **= 2
        total_var = xp.sum(X) / N
    self.explained_variance_ratio_ = self.explained_variance_ / total_var
    self.singular_values_ = xp.asarray(S, copy=True)
    if self.n_components_ < min(n_features, n_samples):
        self.noise_variance_ = total_var - xp.sum(self.explained_variance_)
        self.noise_variance_ /= min(n_features, n_samples) - n_components
    else:
        self.noise_variance_ = 0.0
    return (U, S, Vt)