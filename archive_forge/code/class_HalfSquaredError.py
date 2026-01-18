import numbers
import numpy as np
from scipy.special import xlogy
from ..utils import check_scalar
from ..utils.stats import _weighted_percentile
from ._loss import (
from .link import (
class HalfSquaredError(BaseLoss):
    """Half squared error with identity link, for regression.

    Domain:
    y_true and y_pred all real numbers

    Link:
    y_pred = raw_prediction

    For a given sample x_i, half squared error is defined as::

        loss(x_i) = 0.5 * (y_true_i - raw_prediction_i)**2

    The factor of 0.5 simplifies the computation of gradients and results in a
    unit hessian (and is consistent with what is done in LightGBM). It is also
    half the Normal distribution deviance.
    """

    def __init__(self, sample_weight=None):
        super().__init__(closs=CyHalfSquaredError(), link=IdentityLink())
        self.constant_hessian = sample_weight is None