import torch
from torch import Tensor
from .optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _view_as_real,
from typing import List, Optional
def adagrad(params: List[Tensor], grads: List[Tensor], state_sums: List[Tensor], state_steps: List[Tensor], has_sparse_grad: bool=None, foreach: Optional[bool]=None, differentiable: bool=False, has_complex: bool=False, *, lr: float, weight_decay: float, lr_decay: float, eps: float, maximize: bool):
    """Functional API that performs Adagrad algorithm computation.

    See :class:`~torch.optim.Adagrad` for details.
    """
    if not all((isinstance(t, torch.Tensor) for t in state_steps)):
        raise RuntimeError('API has changed, `state_steps` argument must contain a list of singleton tensors')
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')
    if foreach and (not torch.jit.is_scripting()):
        func = _multi_tensor_adagrad
    else:
        func = _single_tensor_adagrad
    func(params, grads, state_sums, state_steps, lr=lr, weight_decay=weight_decay, lr_decay=lr_decay, eps=eps, has_sparse_grad=has_sparse_grad, maximize=maximize, differentiable=differentiable, has_complex=has_complex)