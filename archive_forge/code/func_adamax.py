import torch
from torch import Tensor
from .optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _stack_if_compiling,
from typing import List, Optional
def adamax(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_infs: List[Tensor], state_steps: List[Tensor], foreach: Optional[bool]=None, maximize: bool=False, differentiable: bool=False, has_complex: bool=False, *, eps: float, beta1: float, beta2: float, lr: float, weight_decay: float):
    """Functional API that performs adamax algorithm computation.

    See :class:`~torch.optim.Adamax` for details.
    """
    if not all((isinstance(t, torch.Tensor) for t in state_steps)):
        raise RuntimeError('API has changed, `state_steps` argument must contain a list of singleton tensors')
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')
    if foreach and (not torch.jit.is_scripting()):
        func = _multi_tensor_adamax
    else:
        func = _single_tensor_adamax
    func(params, grads, exp_avgs, exp_infs, state_steps, eps=eps, beta1=beta1, beta2=beta2, lr=lr, weight_decay=weight_decay, maximize=maximize, differentiable=differentiable, has_complex=has_complex)