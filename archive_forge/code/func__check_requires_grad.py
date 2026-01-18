from typing import List, Tuple
import torch
from torch._vmap_internals import _vmap
from . import forward_ad as fwAD
def _check_requires_grad(inputs, input_type, strict):
    if not strict:
        return
    if input_type not in ['outputs', 'grad_inputs', 'jacobian', 'hessian']:
        raise RuntimeError('Invalid input_type to _check_requires_grad')
    for i, inp in enumerate(inputs):
        if inp is None:
            raise RuntimeError(f'The output of the user-provided function is independent of input {i}. This is not allowed in strict mode.')
        if not inp.requires_grad:
            if input_type == 'hessian':
                raise RuntimeError(f'The hessian of the user-provided function with respect to input {i} is independent of the input. This is not allowed in strict mode. You should ensure that your function is thrice differentiable and that the hessian depends on the inputs.')
            elif input_type == 'jacobian':
                raise RuntimeError(f'While computing the hessian, found that the jacobian of the user-provided function with respect to input {i} is independent of the input. This is not allowed in strict mode. You should ensure that your function is twice differentiable and that the jacobian depends on the inputs (this would be violated by a linear function for example).')
            elif input_type == 'grad_inputs':
                raise RuntimeError(f'The gradient with respect to input {i} is independent of the inputs of the user-provided function. This is not allowed in strict mode.')
            else:
                raise RuntimeError(f'Output {i} of the user-provided function does not require gradients. The outputs must be computed in a differentiable manner from the input when running in strict mode.')