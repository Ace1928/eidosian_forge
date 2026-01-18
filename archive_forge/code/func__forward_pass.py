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
def _forward_pass(self, activations):
    """Perform a forward pass on the network by computing the values
        of the neurons in the hidden layers and the output layer.

        Parameters
        ----------
        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.
        """
    hidden_activation = ACTIVATIONS[self.activation]
    for i in range(self.n_layers_ - 1):
        activations[i + 1] = safe_sparse_dot(activations[i], self.coefs_[i])
        activations[i + 1] += self.intercepts_[i]
        if i + 1 != self.n_layers_ - 1:
            hidden_activation(activations[i + 1])
    output_activation = ACTIVATIONS[self.out_activation_]
    output_activation(activations[i + 1])
    return activations