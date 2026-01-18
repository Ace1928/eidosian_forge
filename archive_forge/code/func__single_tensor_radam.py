import math
from typing import List, Optional
import torch
from torch import Tensor
from .optimizer import (
def _single_tensor_radam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], state_steps: List[Tensor], *, beta1: float, beta2: float, lr: float, weight_decay: float, eps: float, differentiable: bool, decoupled_weight_decay: bool, has_complex: bool):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        if torch.is_complex(param):
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
        step_t += 1
        step = _get_value(step_t)
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        if weight_decay != 0:
            if decoupled_weight_decay:
                param.mul_(1 - lr * weight_decay)
            else:
                grad = grad.add(param, alpha=weight_decay)
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bias_corrected_exp_avg = exp_avg / bias_correction1
        rho_inf = 2 / (1 - beta2) - 1
        rho_t = rho_inf - 2 * step * beta2 ** step / bias_correction2
        if rho_t > 5.0:
            rect = math.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
            exp_avg_sq_sqrt = exp_avg_sq.sqrt()
            if differentiable:
                exp_avg_sq_sqrt = exp_avg_sq_sqrt.add(eps)
            else:
                exp_avg_sq_sqrt = exp_avg_sq_sqrt.add_(eps)
            adaptive_lr = math.sqrt(bias_correction2) / exp_avg_sq_sqrt
            param.add_(bias_corrected_exp_avg * lr * adaptive_lr * rect, alpha=-1.0)
        else:
            param.add_(bias_corrected_exp_avg * lr, alpha=-1.0)