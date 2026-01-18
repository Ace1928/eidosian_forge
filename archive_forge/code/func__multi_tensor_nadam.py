import torch
from torch import Tensor
from .optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _dispatch_sqrt, _stack_if_compiling,
from typing import List, Optional
def _multi_tensor_nadam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], mu_products: List[Tensor], state_steps: List[Tensor], *, beta1: float, beta2: float, lr: float, weight_decay: float, momentum_decay: float, eps: float, decoupled_weight_decay: bool, capturable: bool, differentiable: bool, has_complex: bool):
    if len(params) == 0:
        return
    assert not differentiable, "_foreach ops don't support autograd"
    if not torch._utils.is_compiling() and capturable:
        assert all((p.is_cuda and mp.is_cuda and step.is_cuda for p, mp, step in zip(params, mu_products, state_steps))), 'If capturable=True, params, mu_products, and state_steps must be CUDA tensors.'
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, exp_avgs, exp_avg_sqs, mu_products, state_steps])
    for (grouped_params, grouped_grads, grouped_exp_avgs, grouped_exp_avg_sqs, grouped_mu_products, grouped_state_steps), _ in grouped_tensors.values():
        if has_complex:
            _view_as_real(grouped_params, grouped_grads, grouped_exp_avgs, grouped_exp_avg_sqs)
        if grouped_state_steps[0].is_cpu:
            torch._foreach_add_(grouped_state_steps, torch.tensor(1.0, device='cpu'), alpha=1.0)
        else:
            torch._foreach_add_(grouped_state_steps, 1)
        if weight_decay != 0:
            if decoupled_weight_decay:
                torch._foreach_mul_(grouped_params, 1 - lr * weight_decay)
            else:
                grouped_grads = torch._foreach_add(grouped_grads, grouped_params, alpha=weight_decay)
        torch._foreach_lerp_(grouped_exp_avgs, grouped_grads, 1 - beta1)
        torch._foreach_mul_(grouped_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(grouped_exp_avg_sqs, grouped_grads, grouped_grads, 1 - beta2)
        exp_avg_sq_sqrt = torch._foreach_sqrt(grouped_exp_avg_sqs)
        if capturable:
            exponent = torch._foreach_mul(grouped_state_steps, momentum_decay)
            mus = torch._foreach_pow(0.96, exponent)
            torch._foreach_mul_(mus, -0.5)
            torch._foreach_add_(mus, 1.0)
            torch._foreach_mul_(mus, beta1)
            torch._foreach_add_(exponent, momentum_decay)
            mu_nexts = torch._foreach_pow(0.96, exponent)
            torch._foreach_mul_(mu_nexts, -0.5)
            torch._foreach_add_(mu_nexts, 1.0)
            torch._foreach_mul_(mu_nexts, beta1)
            del exponent
            bias_correction_sqrt = torch._foreach_pow(beta2, grouped_state_steps)
            torch._foreach_sub_(bias_correction_sqrt, 1.0)
            torch._foreach_neg_(bias_correction_sqrt)
            torch._foreach_sqrt_(bias_correction_sqrt)
        else:
            bias_correction_sqrt = [_dispatch_sqrt(1 - beta2 ** _get_value(step)) for step in grouped_state_steps]
            mus = [beta1 * (1.0 - 0.5 * 0.96 ** (_get_value(step) * momentum_decay)) for step in grouped_state_steps]
            mu_nexts = [beta1 * (1.0 - 0.5 * 0.96 ** ((_get_value(step) + 1) * momentum_decay)) for step in grouped_state_steps]
        torch._foreach_mul_(grouped_mu_products, mus)
        torch._foreach_div_(exp_avg_sq_sqrt, bias_correction_sqrt)
        torch._foreach_add_(exp_avg_sq_sqrt, eps)
        del bias_correction_sqrt
        if capturable:
            torch._foreach_sub_(mus, 1.0)
            torch._foreach_mul_(mus, lr)
            denom = torch._foreach_sub(grouped_mu_products, 1.0)
            torch._foreach_neg_(denom)
            torch._foreach_div_(mus, denom)
            step_size_grads = mus
            del denom
            denom = torch._foreach_mul(grouped_mu_products, mu_nexts)
            torch._foreach_mul_(mu_nexts, lr)
            torch._foreach_sub_(denom, 1.0)
            torch._foreach_div_(mu_nexts, denom)
            step_size_expavg = mu_nexts
            del denom
            numerator = torch._foreach_mul(step_size_grads, grouped_grads)
            torch._foreach_addcmul_(numerator, step_size_expavg, grouped_exp_avgs)
            torch._foreach_addcdiv_(grouped_params, numerator, exp_avg_sq_sqrt)
        else:
            step_size_grads = _stack_if_compiling([lr * (1.0 - mu) / (1.0 - _get_value(mu_product)) * -1 for mu_product, mu in zip(grouped_mu_products, mus)])
            step_size_expavg = _stack_if_compiling([lr * mu_next / (1.0 - _get_value(mu_product) * mu_next) * -1 for mu_product, mu_next in zip(grouped_mu_products, mu_nexts)])
            torch._foreach_addcdiv_(grouped_params, grouped_grads, exp_avg_sq_sqrt, step_size_grads)
            torch._foreach_addcdiv_(grouped_params, grouped_exp_avgs, exp_avg_sq_sqrt, step_size_expavg)