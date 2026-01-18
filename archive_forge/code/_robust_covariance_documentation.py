import warnings
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from scipy.stats import chi2
from ..base import _fit_context
from ..utils import check_array, check_random_state
from ..utils._param_validation import Interval
from ..utils.extmath import fast_logdet
from ._empirical_covariance import EmpiricalCovariance, empirical_covariance
Re-weight raw Minimum Covariance Determinant estimates.

        Re-weight observations using Rousseeuw's method (equivalent to
        deleting outlying observations from the data set before
        computing location and covariance estimates) described
        in [RVDriessen]_.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The data matrix, with p features and n samples.
            The data set must be the one which was used to compute
            the raw estimates.

        Returns
        -------
        location_reweighted : ndarray of shape (n_features,)
            Re-weighted robust location estimate.

        covariance_reweighted : ndarray of shape (n_features, n_features)
            Re-weighted robust covariance estimate.

        support_reweighted : ndarray of shape (n_samples,), dtype=bool
            A mask of the observations that have been used to compute
            the re-weighted robust location and covariance estimates.

        References
        ----------

        .. [RVDriessen] A Fast Algorithm for the Minimum Covariance
            Determinant Estimator, 1999, American Statistical Association
            and the American Society for Quality, TECHNOMETRICS
        