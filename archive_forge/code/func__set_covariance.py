import warnings
import numpy as np
from scipy import linalg
from .. import config_context
from ..base import BaseEstimator, _fit_context
from ..metrics.pairwise import pairwise_distances
from ..utils import check_array
from ..utils._param_validation import validate_params
from ..utils.extmath import fast_logdet
def _set_covariance(self, covariance):
    """Saves the covariance and precision estimates

        Storage is done accordingly to `self.store_precision`.
        Precision stored only if invertible.

        Parameters
        ----------
        covariance : array-like of shape (n_features, n_features)
            Estimated covariance matrix to be stored, and from which precision
            is computed.
        """
    covariance = check_array(covariance)
    self.covariance_ = covariance
    if self.store_precision:
        self.precision_ = linalg.pinvh(covariance, check_finite=False)
    else:
        self.precision_ = None