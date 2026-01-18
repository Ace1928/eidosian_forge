import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy.special import xlogy
from ..base import (
from ..metrics import accuracy_score, r2_score
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils import _safe_indexing, check_random_state
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils.extmath import softmax, stable_cumsum
from ..utils.metadata_routing import (
from ..utils.validation import (
from ._base import BaseEnsemble
@staticmethod
def _compute_proba_from_decision(decision, n_classes):
    """Compute probabilities from the decision function.

        This is based eq. (15) of [1] where:
            p(y=c|X) = exp((1 / K-1) f_c(X)) / sum_k(exp((1 / K-1) f_k(X)))
                     = softmax((1 / K-1) * f(X))

        References
        ----------
        .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost",
               2009.
        """
    if n_classes == 2:
        decision = np.vstack([-decision, decision]).T / 2
    else:
        decision /= n_classes - 1
    return softmax(decision, copy=False)