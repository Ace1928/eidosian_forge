import time
import logging
import mxnet as mx
from mxnet.module import Module
from .svrg_optimizer import _SVRGOptimizer
def _svrg_grads_update_rule(self, g_curr_batch_curr_weight, g_curr_batch_special_weight, g_special_weight_all_batch):
    """Calculates the gradient based on the SVRG update rule.
        Parameters
        ----------
        g_curr_batch_curr_weight : NDArray
            gradients of current weight of self.mod w.r.t current batch of data
        g_curr_batch_special_weight: NDArray
            gradients of the weight of past m epochs of self._mod_special w.r.t current batch of data
        g_special_weight_all_batch: NDArray
            average of full gradients over full pass of data

        Returns
        ----------
        Gradients calculated using SVRG update rule:
        grads = g_curr_batch_curr_weight - g_curr_batch_special_weight + g_special_weight_all_batch
        """
    for index, grad in enumerate(g_curr_batch_curr_weight):
        grad -= g_curr_batch_special_weight[index]
        grad += g_special_weight_all_batch[index]
    return g_curr_batch_curr_weight