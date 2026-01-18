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
def _backprop(self, X, y, activations, deltas, coef_grads, intercept_grads):
    """Compute the MLP loss function and its corresponding derivatives
        with respect to each parameter: weights and bias vectors.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,)
            The target values.

        activations : list, length = n_layers - 1
             The ith element of the list holds the values of the ith layer.

        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function

        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.

        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.

        Returns
        -------
        loss : float
        coef_grads : list, length = n_layers - 1
        intercept_grads : list, length = n_layers - 1
        """
    n_samples = X.shape[0]
    activations = self._forward_pass(activations)
    loss_func_name = self.loss
    if loss_func_name == 'log_loss' and self.out_activation_ == 'logistic':
        loss_func_name = 'binary_log_loss'
    loss = LOSS_FUNCTIONS[loss_func_name](y, activations[-1])
    values = 0
    for s in self.coefs_:
        s = s.ravel()
        values += np.dot(s, s)
    loss += 0.5 * self.alpha * values / n_samples
    last = self.n_layers_ - 2
    deltas[last] = activations[-1] - y
    self._compute_loss_grad(last, n_samples, activations, deltas, coef_grads, intercept_grads)
    inplace_derivative = DERIVATIVES[self.activation]
    for i in range(self.n_layers_ - 2, 0, -1):
        deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
        inplace_derivative(activations[i], deltas[i - 1])
        self._compute_loss_grad(i - 1, n_samples, activations, deltas, coef_grads, intercept_grads)
    return (loss, coef_grads, intercept_grads)