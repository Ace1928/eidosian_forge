from numbers import Integral
from operator import itemgetter
import numpy as np
import scipy.optimize
from scipy.linalg import cho_solve, cholesky, solve
from scipy.special import erf, expit
from ..base import BaseEstimator, ClassifierMixin, _fit_context, clone
from ..multiclass import OneVsOneClassifier, OneVsRestClassifier
from ..preprocessing import LabelEncoder
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions
from ..utils.optimize import _check_optimize_result
from ..utils.validation import check_is_fitted
from .kernels import RBF, CompoundKernel, Kernel
from .kernels import ConstantKernel as C
def _posterior_mode(self, K, return_temporaries=False):
    """Mode-finding for binary Laplace GPC and fixed kernel.

        This approximates the posterior of the latent function values for given
        inputs and target observations with a Gaussian approximation and uses
        Newton's iteration to find the mode of this approximation.
        """
    if self.warm_start and hasattr(self, 'f_cached') and (self.f_cached.shape == self.y_train_.shape):
        f = self.f_cached
    else:
        f = np.zeros_like(self.y_train_, dtype=np.float64)
    log_marginal_likelihood = -np.inf
    for _ in range(self.max_iter_predict):
        pi = expit(f)
        W = pi * (1 - pi)
        W_sr = np.sqrt(W)
        W_sr_K = W_sr[:, np.newaxis] * K
        B = np.eye(W.shape[0]) + W_sr_K * W_sr
        L = cholesky(B, lower=True)
        b = W * f + (self.y_train_ - pi)
        a = b - W_sr * cho_solve((L, True), W_sr_K.dot(b))
        f = K.dot(a)
        lml = -0.5 * a.T.dot(f) - np.log1p(np.exp(-(self.y_train_ * 2 - 1) * f)).sum() - np.log(np.diag(L)).sum()
        if lml - log_marginal_likelihood < 1e-10:
            break
        log_marginal_likelihood = lml
    self.f_cached = f
    if return_temporaries:
        return (log_marginal_likelihood, (pi, W_sr, L, b, a))
    else:
        return log_marginal_likelihood