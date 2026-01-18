import torch
from torch import Tensor
from .optimizer import (Optimizer, _default_to_fused_or_foreach, _use_grad_for_differentiable,
from typing import List, Optional
def _multi_tensor_rmsprop(params: List[Tensor], grads: List[Tensor], square_avgs: List[Tensor], grad_avgs: List[Tensor], momentum_buffer_list: List[Tensor], *, lr: float, alpha: float, eps: float, weight_decay: float, momentum: float, centered: bool, maximize: bool, differentiable: bool, has_complex: bool):
    if len(params) == 0:
        return
    assert not differentiable, "_foreach ops don't support autograd"
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, square_avgs, grad_avgs, momentum_buffer_list])
    for (grouped_params, grouped_grads, grouped_square_avgs, grouped_grad_avgs, grouped_momentum_buffer_list), _ in grouped_tensors.values():
        if maximize:
            grouped_grads = torch._foreach_neg(grouped_grads)
        if weight_decay != 0:
            if maximize:
                torch._foreach_add_(grouped_grads, grouped_params, alpha=weight_decay)
            else:
                grouped_grads = torch._foreach_add(grouped_grads, grouped_params, alpha=weight_decay)
        grouped_grads = list(grouped_grads)
        if has_complex:
            state_and_grads = [grouped_grads, grouped_square_avgs]
            if momentum > 0:
                state_and_grads.append(grouped_momentum_buffer_list)
            if centered:
                state_and_grads.append(grouped_grad_avgs)
            _view_as_real(grouped_params, *state_and_grads)
        torch._foreach_mul_(grouped_square_avgs, alpha)
        torch._foreach_addcmul_(grouped_square_avgs, grouped_grads, grouped_grads, value=1 - alpha)
        if centered:
            torch._foreach_lerp_(grouped_grad_avgs, grouped_grads, 1 - alpha)
            avg = torch._foreach_addcmul(grouped_square_avgs, grouped_grad_avgs, grouped_grad_avgs, value=-1)
            torch._foreach_sqrt_(avg)
            torch._foreach_add_(avg, eps)
        else:
            avg = torch._foreach_sqrt(grouped_square_avgs)
            torch._foreach_add_(avg, eps)
        if momentum > 0:
            torch._foreach_mul_(grouped_momentum_buffer_list, momentum)
            torch._foreach_addcdiv_(grouped_momentum_buffer_list, grouped_grads, avg)
            torch._foreach_add_(grouped_params, grouped_momentum_buffer_list, alpha=-lr)
        else:
            torch._foreach_addcdiv_(grouped_params, grouped_grads, avg, value=-lr)