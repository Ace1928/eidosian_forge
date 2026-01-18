import warnings
from numbers import Integral, Real
import numpy as np
import scipy.linalg
from scipy import linalg
from .base import (
from .covariance import empirical_covariance, ledoit_wolf, shrunk_covariance
from .linear_model._base import LinearClassifierMixin
from .preprocessing import StandardScaler
from .utils._array_api import _expit, device, get_namespace, size
from .utils._param_validation import HasMethods, Interval, StrOptions
from .utils.extmath import softmax
from .utils.multiclass import check_classification_targets, unique_labels
from .utils.validation import check_is_fitted
def _solve_eigen(self, X, y, shrinkage, covariance_estimator):
    """Eigenvalue solver.

        The eigenvalue solver computes the optimal solution of the Rayleigh
        coefficient (basically the ratio of between class scatter to within
        class scatter). This solver supports both classification and
        dimensionality reduction (with any covariance estimator).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        shrinkage : 'auto', float or None
            Shrinkage parameter, possible values:
              - None: no shrinkage.
              - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
              - float between 0 and 1: fixed shrinkage constant.

            Shrinkage parameter is ignored if  `covariance_estimator` i
            not None

        covariance_estimator : estimator, default=None
            If not None, `covariance_estimator` is used to estimate
            the covariance matrices instead of relying the empirical
            covariance estimator (with potential shrinkage).
            The object should have a fit method and a ``covariance_`` attribute
            like the estimators in sklearn.covariance.
            if None the shrinkage parameter drives the estimate.

            .. versionadded:: 0.24

        Notes
        -----
        This solver is based on [1]_, section 3.8.3, pp. 121-124.

        References
        ----------
        .. [1] R. O. Duda, P. E. Hart, D. G. Stork. Pattern Classification
           (Second Edition). John Wiley & Sons, Inc., New York, 2001. ISBN
           0-471-05669-3.
        """
    self.means_ = _class_means(X, y)
    self.covariance_ = _class_cov(X, y, self.priors_, shrinkage, covariance_estimator)
    Sw = self.covariance_
    St = _cov(X, shrinkage, covariance_estimator)
    Sb = St - Sw
    evals, evecs = linalg.eigh(Sb, Sw)
    self.explained_variance_ratio_ = np.sort(evals / np.sum(evals))[::-1][:self._max_components]
    evecs = evecs[:, np.argsort(evals)[::-1]]
    self.scalings_ = evecs
    self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
    self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(self.priors_)