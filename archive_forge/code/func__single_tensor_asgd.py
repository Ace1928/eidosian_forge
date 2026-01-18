import torch
from torch import Tensor
from .optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _default_to_fused_or_foreach,
from torch._utils import is_compiling
from typing import List, Optional
def _single_tensor_asgd(params: List[Tensor], grads: List[Tensor], axs: List[Tensor], mus: List[Tensor], etas: List[Tensor], state_steps: List[Tensor], *, lambd: float, lr: float, t0: float, alpha: float, weight_decay: float, maximize: bool, differentiable: bool, capturable: bool, has_complex: bool):
    for i, param in enumerate(params):
        grad = grads[i]
        grad = grad if not maximize else -grad
        mu = mus[i]
        ax = axs[i]
        eta = etas[i]
        step_t = state_steps[i]
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            param = torch.view_as_real(param)
            ax = torch.view_as_real(ax)
        step_t += 1
        step = _get_value(step_t)
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)
        eta_value = _get_value(eta)
        param.mul_(1 - lambd * eta_value)
        param.add_(grad, alpha=-eta_value)
        if is_compiling() or mu.item() != 1:
            ax.add_(param.sub(ax).mul(mu))
        else:
            ax.copy_(param)
        new_eta = _to_tensor(lr / (1 + lambd * lr * step) ** alpha)
        eta.copy_(new_eta)
        new_mu = _to_tensor(1 / max(1, step - t0))
        mu.copy_(new_mu)