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
class DCASGD(Optimizer):
    """The DCASGD optimizer.

    This class implements the optimizer described in *Asynchronous Stochastic Gradient Descent
    with Delay Compensation for Distributed Deep Learning*,
    available at https://arxiv.org/abs/1609.08326.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    momentum : float, optional
       The momentum value.

    lamda : float, optional
       Scale DC value.
    """

    def __init__(self, momentum=0.0, lamda=0.04, **kwargs):
        super(DCASGD, self).__init__(**kwargs)
        self.momentum = momentum
        self.weight_previous = {}
        self.lamda = lamda

    def create_state(self, index, weight):
        if self.momentum == 0.0:
            return (None, weight.copy())
        else:
            return (zeros(weight.shape, weight.context, dtype=weight.dtype), weight.copy())

    def update(self, index, weight, grad, state):
        assert isinstance(weight, NDArray)
        assert isinstance(grad, NDArray)
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)
        mom, previous_weight = state
        if mom:
            mom[:] *= self.momentum
            mom[:] += -lr * (grad + wd * weight + self.lamda * grad * grad * (weight - previous_weight))
        else:
            assert self.momentum == 0.0
            mom = -lr * (grad + wd * weight + self.lamda * grad * grad * (weight - previous_weight))
        previous_weight[:] = weight
        weight[:] += mom