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
def _solve_lstsq(self, X, y, shrinkage, covariance_estimator):
    """Least squares solver.

        The least squares solver computes a straightforward solution of the
        optimal decision rule based directly on the discriminant functions. It
        can only be used for classification (with any covariance estimator),
        because
        estimation of eigenvectors is not performed. Therefore, dimensionality
        reduction with the transform is not supported.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_classes)
            Target values.

        shrinkage : 'auto', float or None
            Shrinkage parameter, possible values:
              - None: no shrinkage.
              - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
              - float between 0 and 1: fixed shrinkage parameter.

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
        This solver is based on [1]_, section 2.6.2, pp. 39-41.

        References
        ----------
        .. [1] R. O. Duda, P. E. Hart, D. G. Stork. Pattern Classification
           (Second Edition). John Wiley & Sons, Inc., New York, 2001. ISBN
           0-471-05669-3.
        """
    self.means_ = _class_means(X, y)
    self.covariance_ = _class_cov(X, y, self.priors_, shrinkage, covariance_estimator)
    self.coef_ = linalg.lstsq(self.covariance_, self.means_.T)[0].T
    self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(self.priors_)