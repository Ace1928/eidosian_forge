import numbers
import numpy as np
from scipy.special import xlogy
from ..utils import check_scalar
from ..utils.stats import _weighted_percentile
from ._loss import (
from .link import (
class HalfTweedieLoss(BaseLoss):
    """Half Tweedie deviance loss with log-link, for regression.

    Domain:
    y_true in real numbers for power <= 0
    y_true in non-negative real numbers for 0 < power < 2
    y_true in positive real numbers for 2 <= power
    y_pred in positive real numbers
    power in real numbers

    Link:
    y_pred = exp(raw_prediction)

    For a given sample x_i, half Tweedie deviance loss with p=power is defined
    as::

        loss(x_i) = max(y_true_i, 0)**(2-p) / (1-p) / (2-p)
                    - y_true_i * exp(raw_prediction_i)**(1-p) / (1-p)
                    + exp(raw_prediction_i)**(2-p) / (2-p)

    Taking the limits for p=0, 1, 2 gives HalfSquaredError with a log link,
    HalfPoissonLoss and HalfGammaLoss.

    We also skip constant terms, but those are different for p=0, 1, 2.
    Therefore, the loss is not continuous in `power`.

    Note furthermore that although no Tweedie distribution exists for
    0 < power < 1, it still gives a strictly consistent scoring function for
    the expectation.
    """

    def __init__(self, sample_weight=None, power=1.5):
        super().__init__(closs=CyHalfTweedieLoss(power=float(power)), link=LogLink())
        if self.closs.power <= 0:
            self.interval_y_true = Interval(-np.inf, np.inf, False, False)
        elif self.closs.power < 2:
            self.interval_y_true = Interval(0, np.inf, True, False)
        else:
            self.interval_y_true = Interval(0, np.inf, False, False)

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        if self.closs.power == 0:
            return HalfSquaredError().constant_to_optimal_zero(y_true=y_true, sample_weight=sample_weight)
        elif self.closs.power == 1:
            return HalfPoissonLoss().constant_to_optimal_zero(y_true=y_true, sample_weight=sample_weight)
        elif self.closs.power == 2:
            return HalfGammaLoss().constant_to_optimal_zero(y_true=y_true, sample_weight=sample_weight)
        else:
            p = self.closs.power
            term = np.power(np.maximum(y_true, 0), 2 - p) / (1 - p) / (2 - p)
            if sample_weight is not None:
                term *= sample_weight
            return term