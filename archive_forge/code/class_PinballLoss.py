import numbers
import numpy as np
from scipy.special import xlogy
from ..utils import check_scalar
from ..utils.stats import _weighted_percentile
from ._loss import (
from .link import (
class PinballLoss(BaseLoss):
    """Quantile loss aka pinball loss, for regression.

    Domain:
    y_true and y_pred all real numbers
    quantile in (0, 1)

    Link:
    y_pred = raw_prediction

    For a given sample x_i, the pinball loss is defined as::

        loss(x_i) = rho_{quantile}(y_true_i - raw_prediction_i)

        rho_{quantile}(u) = u * (quantile - 1_{u<0})
                          = -u *(1 - quantile)  if u < 0
                             u * quantile       if u >= 0

    Note: 2 * PinballLoss(quantile=0.5) equals AbsoluteError().

    Note that the exact hessian = 0 almost everywhere (except at one point, therefore
    differentiable = False). Optimization routines like in HGBT, however, need a
    hessian > 0. Therefore, we assign 1.

    Additional Attributes
    ---------------------
    quantile : float
        The quantile level of the quantile to be estimated. Must be in range (0, 1).
    """
    differentiable = False
    need_update_leaves_values = True

    def __init__(self, sample_weight=None, quantile=0.5):
        check_scalar(quantile, 'quantile', target_type=numbers.Real, min_val=0, max_val=1, include_boundaries='neither')
        super().__init__(closs=CyPinballLoss(quantile=float(quantile)), link=IdentityLink())
        self.approx_hessian = True
        self.constant_hessian = sample_weight is None

    def fit_intercept_only(self, y_true, sample_weight=None):
        """Compute raw_prediction of an intercept-only model.

        This is the weighted median of the target, i.e. over the samples
        axis=0.
        """
        if sample_weight is None:
            return np.percentile(y_true, 100 * self.closs.quantile, axis=0)
        else:
            return _weighted_percentile(y_true, sample_weight, 100 * self.closs.quantile)