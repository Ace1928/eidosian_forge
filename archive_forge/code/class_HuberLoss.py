import numbers
import numpy as np
from scipy.special import xlogy
from ..utils import check_scalar
from ..utils.stats import _weighted_percentile
from ._loss import (
from .link import (
class HuberLoss(BaseLoss):
    """Huber loss, for regression.

    Domain:
    y_true and y_pred all real numbers
    quantile in (0, 1)

    Link:
    y_pred = raw_prediction

    For a given sample x_i, the Huber loss is defined as::

        loss(x_i) = 1/2 * abserr**2            if abserr <= delta
                    delta * (abserr - delta/2) if abserr > delta

        abserr = |y_true_i - raw_prediction_i|
        delta = quantile(abserr, self.quantile)

    Note: HuberLoss(quantile=1) equals HalfSquaredError and HuberLoss(quantile=0)
    equals delta * (AbsoluteError() - delta/2).

    Additional Attributes
    ---------------------
    quantile : float
        The quantile level which defines the breaking point `delta` to distinguish
        between absolute error and squared error. Must be in range (0, 1).

     Reference
    ---------
    .. [1] Friedman, J.H. (2001). :doi:`Greedy function approximation: A gradient
      boosting machine <10.1214/aos/1013203451>`.
      Annals of Statistics, 29, 1189-1232.
    """
    differentiable = False
    need_update_leaves_values = True

    def __init__(self, sample_weight=None, quantile=0.9, delta=0.5):
        check_scalar(quantile, 'quantile', target_type=numbers.Real, min_val=0, max_val=1, include_boundaries='neither')
        self.quantile = quantile
        super().__init__(closs=CyHuberLoss(delta=float(delta)), link=IdentityLink())
        self.approx_hessian = True
        self.constant_hessian = False

    def fit_intercept_only(self, y_true, sample_weight=None):
        """Compute raw_prediction of an intercept-only model.

        This is the weighted median of the target, i.e. over the samples
        axis=0.
        """
        if sample_weight is None:
            median = np.percentile(y_true, 50, axis=0)
        else:
            median = _weighted_percentile(y_true, sample_weight, 50)
        diff = y_true - median
        term = np.sign(diff) * np.minimum(self.closs.delta, np.abs(diff))
        return median + np.average(term, weights=sample_weight)