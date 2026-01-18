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
Return log-marginal likelihood of theta for training data.

        In the case of multi-class classification, the mean log-marginal
        likelihood of the one-versus-rest classifiers are returned.

        Parameters
        ----------
        theta : array-like of shape (n_kernel_params,), default=None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. In the case of multi-class classification, theta may
            be the  hyperparameters of the compound kernel or of an individual
            kernel. In the latter case, all individual kernel get assigned the
            same theta values. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.

        eval_gradient : bool, default=False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. Note that gradient computation is not supported
            for non-binary classification. If True, theta must not be None.

        clone_kernel : bool, default=True
            If True, the kernel attribute is copied. If False, the kernel
            attribute is modified, but may result in a performance improvement.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.

        log_likelihood_gradient : ndarray of shape (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when `eval_gradient` is True.
        