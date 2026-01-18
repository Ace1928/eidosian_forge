import logging
import math
import pickle
import warnings
import os
import numpy
from ..base import py_str
from ..ndarray import (NDArray, zeros, clip, sqrt, cast, maximum, abs as NDabs, array, multiply,
from ..ndarray import (sgd_update, sgd_mom_update, adam_update, rmsprop_update, rmspropalex_update,
from ..ndarray.contrib import (multi_lamb_update, multi_mp_lamb_update)
from ..ndarray import sparse
from ..random import normal
from ..util import is_np_array
@register
class AdaDelta(Optimizer):
    """The AdaDelta optimizer.

    This class implements AdaDelta, an optimizer described in  *ADADELTA: An adaptive
    learning rate method*, available at https://arxiv.org/abs/1212.5701.

    This optimizer updates each weight by::

        grad = clip(grad * rescale_grad + wd * weight, clip_gradient)
        acc_grad = rho * acc_grad + (1. - rho) * grad * grad
        delta = sqrt(acc_delta + epsilon) / sqrt(acc_grad + epsilon) * grad
        acc_delta = rho * acc_delta + (1. - rho) * delta * delta
        weight -= (delta + wd * weight)

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    rho: float
        Decay rate for both squared gradients and delta.
    epsilon : float
        Small value to avoid division by 0.
    """

    def __init__(self, rho=0.9, epsilon=1e-05, **kwargs):
        super(AdaDelta, self).__init__(**kwargs)
        self.rho = rho
        self.epsilon = epsilon

    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context), zeros(weight.shape, weight.context))

    def update(self, index, weight, grad, state):
        assert isinstance(weight, NDArray)
        assert isinstance(grad, NDArray)
        wd = self._get_wd(index)
        self._update_count(index)
        grad *= self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)
        acc_g, acc_delta = state
        acc_g[:] *= self.rho
        acc_g[:] += (1.0 - self.rho) * grad * grad
        current_delta = sqrt(acc_delta + self.epsilon) / sqrt(acc_g + self.epsilon) * grad
        acc_delta[:] *= self.rho
        acc_delta[:] += (1.0 - self.rho) * current_delta * current_delta
        weight[:] -= current_delta + wd * weight