import torch
from torch import Tensor
from .optimizer import (Optimizer, _use_grad_for_differentiable, _default_to_fused_or_foreach,
from typing import List, Optional
def _multi_tensor_rprop(params: List[Tensor], grads: List[Tensor], prevs: List[Tensor], step_sizes: List[Tensor], *, step_size_min: float, step_size_max: float, etaminus: float, etaplus: float, maximize: bool, differentiable: bool, has_complex: bool):
    if len(params) == 0:
        return
    assert not differentiable, "_foreach ops don't support autograd"
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, prevs, step_sizes])
    for (grouped_params, grouped_grads, grouped_prevs, grouped_step_sizes), _ in grouped_tensors.values():
        if has_complex:
            _view_as_real(grouped_params, grouped_grads, grouped_prevs, grouped_step_sizes)
        signs = torch._foreach_mul(grouped_grads, grouped_prevs)
        if maximize:
            torch._foreach_neg_(signs)
        torch._foreach_copy_(grouped_prevs, grouped_grads)
        if maximize:
            torch._foreach_neg_(grouped_prevs)
        grouped_grads = grouped_prevs
        torch._foreach_sign_(signs)
        for sign in signs:
            sign[sign.gt(0)] = etaplus
            sign[sign.lt(0)] = etaminus
            sign[sign.eq(0)] = 1
        torch._foreach_mul_(grouped_step_sizes, signs)
        for step_size in grouped_step_sizes:
            step_size.clamp_(step_size_min, step_size_max)
        grouped_grads = list(grouped_grads)
        for i in range(len(grouped_grads)):
            grouped_grads[i][signs[i].eq(etaminus)] = 0
        del signs
        grad_signs = [grad.sign() for grad in grouped_grads]
        torch._foreach_addcmul_(grouped_params, grad_signs, grouped_step_sizes, value=-1)