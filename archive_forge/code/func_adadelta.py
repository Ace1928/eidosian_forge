import torch
from torch import Tensor
from .optimizer import (Optimizer, _use_grad_for_differentiable, _default_to_fused_or_foreach,
from typing import List, Optional
def adadelta(params: List[Tensor], grads: List[Tensor], square_avgs: List[Tensor], acc_deltas: List[Tensor], foreach: Optional[bool]=None, differentiable: bool=False, has_complex: bool=False, *, lr: float, rho: float, eps: float, weight_decay: float, maximize: bool):
    """Functional API that performs Adadelta algorithm computation.

    See :class:`~torch.optim.Adadelta` for details.
    """
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')
    if foreach and (not torch.jit.is_scripting()):
        func = _multi_tensor_adadelta
    else:
        func = _single_tensor_adadelta
    func(params, grads, square_avgs, acc_deltas, lr=lr, rho=rho, eps=eps, weight_decay=weight_decay, maximize=maximize, differentiable=differentiable, has_complex=has_complex)