import warnings
from abc import ABCMeta, abstractmethod
from itertools import chain
from numbers import Integral, Real
import numpy as np
import scipy.optimize
from ..base import (
from ..exceptions import ConvergenceWarning
from ..metrics import accuracy_score, r2_score
from ..model_selection import train_test_split
from ..preprocessing import LabelBinarizer
from ..utils import (
from ..utils._param_validation import Interval, Options, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.metaestimators import available_if
from ..utils.multiclass import (
from ..utils.optimize import _check_optimize_result
from ..utils.validation import check_is_fitted
from ._base import ACTIVATIONS, DERIVATIVES, LOSS_FUNCTIONS
from ._stochastic_optimizers import AdamOptimizer, SGDOptimizer
def _compute_loss_grad(self, layer, n_samples, activations, deltas, coef_grads, intercept_grads):
    """Compute the gradient of loss with respect to coefs and intercept for
        specified layer.

        This function does backpropagation for the specified one layer.
        """
    coef_grads[layer] = safe_sparse_dot(activations[layer].T, deltas[layer])
    coef_grads[layer] += self.alpha * self.coefs_[layer]
    coef_grads[layer] /= n_samples
    intercept_grads[layer] = np.mean(deltas[layer], 0)