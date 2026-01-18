import torch
from torch import Tensor
from .optimizer import (Optimizer, _default_to_fused_or_foreach, _use_grad_for_differentiable,
from typing import List, Optional
def _single_tensor_rmsprop(params: List[Tensor], grads: List[Tensor], square_avgs: List[Tensor], grad_avgs: List[Tensor], momentum_buffer_list: List[Tensor], *, lr: float, alpha: float, eps: float, weight_decay: float, momentum: float, centered: bool, maximize: bool, differentiable: bool, has_complex: bool):
    for i, param in enumerate(params):
        grad = grads[i]
        grad = grad if not maximize else -grad
        square_avg = square_avgs[i]
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)
        is_complex_param = torch.is_complex(param)
        if is_complex_param:
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            square_avg = torch.view_as_real(square_avg)
        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
        if centered:
            grad_avg = grad_avgs[i]
            if is_complex_param:
                grad_avg = torch.view_as_real(grad_avg)
            grad_avg.lerp_(grad, 1 - alpha)
            avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_()
        else:
            avg = square_avg.sqrt()
        if differentiable:
            avg = avg.add(eps)
        else:
            avg = avg.add_(eps)
        if momentum > 0:
            buf = momentum_buffer_list[i]
            if is_complex_param:
                buf = torch.view_as_real(buf)
            buf.mul_(momentum).addcdiv_(grad, avg)
            param.add_(buf, alpha=-lr)
        else:
            param.addcdiv_(grad, avg, value=-lr)