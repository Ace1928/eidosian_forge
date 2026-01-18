import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type
import numpy as np
import torch
from torch.autograd import Variable
import torch.distributed as dist
from torch.optim import SGD, Optimizer
def _final_callback(self) -> None:
    self._final_callback_queued = False
    assert isinstance(self._local_grad_sqr, torch.Tensor)
    self._num_backward_calls += 1
    assert self._num_backward_calls - self._last_final_backward_call <= self._num_grads_to_accum, f'bug: {self._num_backward_calls} - {self._last_final_backward_call} should <= {self._num_grads_to_accum}'
    if (self._num_backward_calls - self._last_final_backward_call) % self._num_grads_to_accum != 0:
        assert self._local_grad_sqr is not None, 'We should still be in backward phase'
        return
    work = None
    if self._world_size > 1:
        work = dist.all_reduce(self._local_grad_sqr, async_op=True)
    total_grad_sqr = np.array([sum((param.grad.pow(2).sum().item() for param in group['params'])) for group in self._optimizer.param_groups])
    if work:
        work.wait()
    local_grad_sqr = self._local_grad_sqr.cpu().numpy()
    if self._num_grads_to_accum > 1:
        if self._is_scaled_loss:
            local_grad_sqr *= self._num_grads_to_accum ** 2
        else:
            total_grad_sqr /= self._num_grads_to_accum ** 2
    S = self._scale
    cN = self._world_size * self._num_grads_to_accum
    grad_var = local_grad_sqr * (S / cN) / (cN - 1) - total_grad_sqr * S / (cN - 1)
    grad_sqr = total_grad_sqr - grad_var / S
    grad_var = np.maximum(grad_var, 1e-06)
    grad_sqr = np.maximum(grad_sqr, 0.0)
    self._update_avg('grad_sqr_avg', grad_sqr, self.smoothing)
    self._update_avg('grad_var_avg', grad_var, self.smoothing)
    self._last_final_backward_call = self._num_backward_calls
    self._local_grad_sqr = None