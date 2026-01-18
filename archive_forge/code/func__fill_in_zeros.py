from typing import List, Tuple
import torch
from torch._vmap_internals import _vmap
from . import forward_ad as fwAD
def _fill_in_zeros(grads, refs, strict, create_graph, stage):
    if stage not in ['back', 'back_trick', 'double_back', 'double_back_trick']:
        raise RuntimeError(f"Invalid stage argument '{stage}' to _fill_in_zeros")
    res: Tuple[torch.Tensor, ...] = tuple()
    for i, grads_i in enumerate(grads):
        if grads_i is None:
            if strict:
                if stage == 'back':
                    raise RuntimeError(f'The output of the user-provided function is independent of input {i}. This is not allowed in strict mode.')
                elif stage == 'back_trick':
                    raise RuntimeError(f'The gradient with respect to the input is independent of entry {i} in the grad_outputs when using the double backward trick to compute forward mode gradients. This is not allowed in strict mode.')
                elif stage == 'double_back':
                    raise RuntimeError(f'The jacobian of the user-provided function is independent of input {i}. This is not allowed in strict mode.')
                else:
                    raise RuntimeError(f'The hessian of the user-provided function is independent of entry {i} in the grad_jacobian. This is not allowed in strict mode as it prevents from using the double backward trick to replace forward mode AD.')
            grads_i = torch.zeros_like(refs[i])
        elif strict and create_graph and (not grads_i.requires_grad):
            if 'double' not in stage:
                raise RuntimeError(f'The jacobian of the user-provided function is independent of input {i}. This is not allowed in strict mode when create_graph=True.')
            else:
                raise RuntimeError(f'The hessian of the user-provided function is independent of input {i}. This is not allowed in strict mode when create_graph=True.')
        res += (grads_i,)
    return res