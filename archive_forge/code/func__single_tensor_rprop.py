import torch
from torch import Tensor
from .optimizer import (Optimizer, _use_grad_for_differentiable, _default_to_fused_or_foreach,
from typing import List, Optional
def _single_tensor_rprop(params: List[Tensor], grads: List[Tensor], prevs: List[Tensor], step_sizes: List[Tensor], *, step_size_min: float, step_size_max: float, etaminus: float, etaplus: float, maximize: bool, differentiable: bool, has_complex: bool):
    for i, param in enumerate(params):
        grad = grads[i]
        grad = grad if not maximize else -grad
        prev = prevs[i]
        step_size = step_sizes[i]
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            prev = torch.view_as_real(prev)
            param = torch.view_as_real(param)
            step_size = torch.view_as_real(step_size)
        if differentiable:
            sign = grad.mul(prev.clone()).sign()
        else:
            sign = grad.mul(prev).sign()
        sign[sign.gt(0)] = etaplus
        sign[sign.lt(0)] = etaminus
        sign[sign.eq(0)] = 1
        step_size.mul_(sign).clamp_(step_size_min, step_size_max)
        grad = grad.clone(memory_format=torch.preserve_format)
        grad[sign.eq(etaminus)] = 0
        param.addcmul_(grad.sign(), step_size, value=-1)
        prev.copy_(grad)