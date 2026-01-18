from typing import List, Optional, Union, Tuple
import torch
from torch import Tensor
from .optimizer import (Optimizer, ParamsT, _use_grad_for_differentiable, _get_value,
from torch.utils._foreach_utils import _get_fused_kernels_supported_devices
def _multi_tensor_adam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], max_exp_avg_sqs: List[Tensor], state_steps: List[Tensor], grad_scale: Optional[Tensor], found_inf: Optional[Tensor], *, amsgrad: bool, has_complex: bool, beta1: float, beta2: float, lr: Union[float, Tensor], weight_decay: float, eps: float, maximize: bool, capturable: bool, differentiable: bool):
    if len(params) == 0:
        return
    if isinstance(lr, Tensor) and (not capturable):
        raise RuntimeError('lr as a Tensor is not supported for capturable=False and foreach=True')
    if not torch._utils.is_compiling() and capturable:
        assert all((p.is_cuda and step.is_cuda for p, step in zip(params, state_steps))), 'If capturable=True, params and state_steps must be CUDA tensors.'
    assert grad_scale is None and found_inf is None
    assert not differentiable, "_foreach ops don't support autograd"
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps])
    for (device_params, device_grads, device_exp_avgs, device_exp_avg_sqs, device_max_exp_avg_sqs, device_state_steps), _ in grouped_tensors.values():
        if maximize:
            device_grads = torch._foreach_neg(device_grads)
        if has_complex:
            if amsgrad:
                _view_as_real(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs, device_max_exp_avg_sqs)
            else:
                _view_as_real(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs)
        if device_state_steps[0].is_cpu:
            torch._foreach_add_(device_state_steps, torch.tensor(1.0, device='cpu'), alpha=1.0)
        else:
            torch._foreach_add_(device_state_steps, 1)
        if weight_decay != 0:
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(device_grads, device_params, alpha=weight_decay)
        torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)
        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads, 1 - beta2)
        del device_grads
        if capturable:
            bias_correction1 = torch._foreach_pow(beta1, device_state_steps)
            bias_correction2 = torch._foreach_pow(beta2, device_state_steps)
            torch._foreach_sub_(bias_correction1, 1)
            torch._foreach_sub_(bias_correction2, 1)
            torch._foreach_neg_(bias_correction2)
            torch._foreach_div_(bias_correction1, lr)
            torch._foreach_reciprocal_(bias_correction1)
            torch._foreach_sqrt_(bias_correction2)
            step_size = bias_correction1
            bias_correction2_sqrt = bias_correction2
            if amsgrad:
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_div_(exp_avg_sq_sqrt, step_size)
            torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt)
        else:
            bias_correction1 = [1 - beta1 ** _get_value(step) for step in device_state_steps]
            bias_correction2 = [1 - beta2 ** _get_value(step) for step in device_state_steps]
            step_size = _stack_if_compiling([lr / bc * -1 for bc in bias_correction1])
            bias_correction2_sqrt = [_dispatch_sqrt(bc) for bc in bias_correction2]
            if amsgrad:
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt, step_size)