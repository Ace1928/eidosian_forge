import torch
from torch import Tensor
from .optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _dispatch_sqrt,
from typing import List, Optional, Tuple, Union
from torch.utils._foreach_utils import _get_fused_kernels_supported_devices
def adamw(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], max_exp_avg_sqs: List[Tensor], state_steps: List[Tensor], foreach: Optional[bool]=None, capturable: bool=False, differentiable: bool=False, fused: Optional[bool]=None, grad_scale: Optional[Tensor]=None, found_inf: Optional[Tensor]=None, has_complex: bool=False, *, amsgrad: bool, beta1: float, beta2: float, lr: Union[float, Tensor], weight_decay: float, eps: float, maximize: bool):
    """Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """
    if not torch._utils.is_compiling() and (not all((isinstance(t, torch.Tensor) for t in state_steps))):
        raise RuntimeError('API has changed, `state_steps` argument must contain a list of singleton tensors')
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
        if foreach and isinstance(lr, Tensor) and (not capturable):
            foreach = False
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False
    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')
    if fused and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with fused optimizers')
    if fused and (not torch.jit.is_scripting()):
        func = _fused_adamw
    elif foreach and (not torch.jit.is_scripting()):
        func = _multi_tensor_adamw
    else:
        func = _single_tensor_adamw
    func(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, amsgrad=amsgrad, beta1=beta1, beta2=beta2, lr=lr, weight_decay=weight_decay, eps=eps, maximize=maximize, capturable=capturable, differentiable=differentiable, grad_scale=grad_scale, found_inf=found_inf, has_complex=has_complex)