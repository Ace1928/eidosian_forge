import math
from typing import List, Optional
import torch
from torch import Tensor
from .optimizer import (
def _multi_tensor_radam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], state_steps: List[Tensor], *, beta1: float, beta2: float, lr: float, weight_decay: float, eps: float, decoupled_weight_decay: bool, differentiable: bool, has_complex: bool):
    if len(params) == 0:
        return
    assert not differentiable, "_foreach ops don't support autograd"
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, exp_avgs, exp_avg_sqs, state_steps])
    for (grouped_params, grouped_grads, grouped_exp_avgs, grouped_exp_avg_sqs, grouped_state_steps), _ in grouped_tensors.values():
        if grouped_state_steps[0].is_cpu:
            torch._foreach_add_(grouped_state_steps, torch.tensor(1.0, device='cpu'), alpha=1.0)
        else:
            torch._foreach_add_(grouped_state_steps, 1)
        if has_complex:
            _view_as_real(grouped_params, grouped_grads, grouped_exp_avgs, grouped_exp_avg_sqs)
        rho_inf = 2 / (1 - beta2) - 1
        rho_t_list = [rho_inf - 2 * _get_value(step) * beta2 ** _get_value(step) / (1 - beta2 ** _get_value(step)) for step in grouped_state_steps]
        if weight_decay != 0:
            if decoupled_weight_decay:
                torch._foreach_mul_(grouped_params, 1 - lr * weight_decay)
            else:
                grouped_grads = torch._foreach_add(grouped_grads, grouped_params, alpha=weight_decay)
        torch._foreach_lerp_(grouped_exp_avgs, grouped_grads, 1 - beta1)
        torch._foreach_mul_(grouped_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(grouped_exp_avg_sqs, grouped_grads, grouped_grads, 1 - beta2)
        del grouped_grads
        rect = [_dispatch_sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t)) if rho_t > 5 else 0 for rho_t in rho_t_list]
        unrectified = [0 if rect > 0 else 1.0 for rect in rect]
        bias_correction1 = [1 - beta1 ** _get_value(step) for step in grouped_state_steps]
        unrect_step_size = _stack_if_compiling([lr * rect / bc * -1 for rect, bc in zip(unrectified, bias_correction1)])
        bias_correction2_sqrt_times_rect_step_size = [_dispatch_sqrt(1 - beta2 ** _get_value(step)) * (lr * rect / bc) * -1 for step, rect, bc in zip(grouped_state_steps, rect, bias_correction1)]
        buffer = torch._foreach_sqrt(grouped_exp_avg_sqs)
        torch._foreach_add_(buffer, eps)
        torch._foreach_div_(buffer, bias_correction2_sqrt_times_rect_step_size)
        torch._foreach_reciprocal_(buffer)
        torch._foreach_add_(buffer, unrect_step_size)
        torch._foreach_addcmul_(grouped_params, grouped_exp_avgs, buffer)