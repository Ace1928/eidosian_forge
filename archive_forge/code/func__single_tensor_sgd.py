import torch
from torch import Tensor
from .optimizer import (Optimizer, _use_grad_for_differentiable, _default_to_fused_or_foreach,
from typing import List, Optional
def _single_tensor_sgd(params: List[Tensor], d_p_list: List[Tensor], momentum_buffer_list: List[Optional[Tensor]], *, weight_decay: float, momentum: float, lr: float, dampening: float, nesterov: bool, maximize: bool, has_sparse_grad: bool):
    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)
        if momentum != 0:
            buf = momentum_buffer_list[i]
            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf
        param.add_(d_p, alpha=-lr)