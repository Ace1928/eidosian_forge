import numbers
import numpy as np
from scipy.special import xlogy
from ..utils import check_scalar
from ..utils.stats import _weighted_percentile
from ._loss import (
from .link import (
class ExponentialLoss(BaseLoss):
    """Exponential loss with (half) logit link, for binary classification.

    This is also know as boosting loss.

    Domain:
    y_true in [0, 1], i.e. regression on the unit interval
    y_pred in (0, 1), i.e. boundaries excluded

    Link:
    y_pred = expit(2 * raw_prediction)

    For a given sample x_i, the exponential loss is defined as::

        loss(x_i) = y_true_i * exp(-raw_pred_i)) + (1 - y_true_i) * exp(raw_pred_i)

    See:
    - J. Friedman, T. Hastie, R. Tibshirani.
      "Additive logistic regression: a statistical view of boosting (With discussion
      and a rejoinder by the authors)." Ann. Statist. 28 (2) 337 - 407, April 2000.
      https://doi.org/10.1214/aos/1016218223
    - A. Buja, W. Stuetzle, Y. Shen. (2005).
      "Loss Functions for Binary Class Probability Estimation and Classification:
      Structure and Applications."

    Note that the formulation works for classification, y = {0, 1}, as well as
    "exponential logistic" regression, y = [0, 1].
    Note that this is a proper scoring rule, but without it's canonical link.

    More details: Inserting the predicted probability
    y_pred = expit(2 * raw_prediction) in the loss gives::

        loss(x_i) = y_true_i * sqrt((1 - y_pred_i) / y_pred_i)
            + (1 - y_true_i) * sqrt(y_pred_i / (1 - y_pred_i))
    """

    def __init__(self, sample_weight=None):
        super().__init__(closs=CyExponentialLoss(), link=HalfLogitLink(), n_classes=2)
        self.interval_y_true = Interval(0, 1, True, True)

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        term = -2 * np.sqrt(y_true * (1 - y_true))
        if sample_weight is not None:
            term *= sample_weight
        return term

    def predict_proba(self, raw_prediction):
        """Predict probabilities.

        Parameters
        ----------
        raw_prediction : array of shape (n_samples,) or (n_samples, 1)
            Raw prediction values (in link space).

        Returns
        -------
        proba : array of shape (n_samples, 2)
            Element-wise class probabilities.
        """
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)
        proba = np.empty((raw_prediction.shape[0], 2), dtype=raw_prediction.dtype)
        proba[:, 1] = self.link.inverse(raw_prediction)
        proba[:, 0] = 1 - proba[:, 1]
        return proba