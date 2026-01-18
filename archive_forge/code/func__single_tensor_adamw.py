import torch
from torch import Tensor
from .optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _dispatch_sqrt,
from typing import List, Optional, Tuple, Union
from torch.utils._foreach_utils import _get_fused_kernels_supported_devices
def _single_tensor_adamw(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], max_exp_avg_sqs: List[Tensor], state_steps: List[Tensor], grad_scale: Optional[Tensor], found_inf: Optional[Tensor], *, amsgrad: bool, beta1: float, beta2: float, lr: Union[Tensor, float], weight_decay: float, eps: float, maximize: bool, capturable: bool, differentiable: bool, has_complex: bool):
    assert grad_scale is None and found_inf is None
    if torch.jit.is_scripting():
        assert isinstance(lr, float)
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        if not torch._utils.is_compiling() and capturable:
            assert param.is_cuda and step_t.is_cuda or (param.is_xla and step_t.is_xla), 'If capturable=True, params and state_steps must be CUDA or XLA tensors.'
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
            param = torch.view_as_real(param)
        step_t += 1
        param.mul_(1 - lr * weight_decay)
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if capturable or differentiable:
            step = step_t
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()
            bias_correction2_sqrt = bias_correction2.sqrt()
            if amsgrad:
                if differentiable:
                    max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sq = max_exp_avg_sqs[i]
                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))
                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            else:
                denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            param.addcdiv_(exp_avg, denom)
        else:
            step = _get_value(step_t)
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1
            bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)
            if amsgrad:
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            param.addcdiv_(exp_avg, denom, value=-step_size)
        if amsgrad and torch.is_complex(params[i]):
            max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])