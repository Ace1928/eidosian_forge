import collections
import functools
import warnings
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.testing
from torch._vmap_internals import _vmap, vmap
from torch.overrides import is_tensor_like
from torch.types import _TensorOrTensors
def _check_analytical_jacobian_attributes(inputs, output, nondet_tol, check_grad_dtypes, fast_mode=False, v=None) -> Tuple[torch.Tensor, ...]:
    diff_input_list = list(_iter_tensors(inputs, True))

    def vjp_fn(grad_output):
        return torch.autograd.grad(output, diff_input_list, grad_output, retain_graph=True, allow_unused=True)
    if fast_mode:
        vjps1 = _get_analytical_vjps_wrt_specific_output(vjp_fn, output.clone(), v)
        vjps2 = _get_analytical_vjps_wrt_specific_output(vjp_fn, output.clone(), v)
    else:
        vjps1 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())
        vjps2 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())
    output_numel = output.numel() if not fast_mode else 1
    jacobians1, types_ok, sizes_ok = _stack_and_check_tensors(vjps1, inputs, output_numel)
    jacobians2, _, _ = _stack_and_check_tensors(vjps2, inputs, output_numel)
    reentrant = _check_jacobians_equal(jacobians1, jacobians2, nondet_tol)
    if not types_ok and check_grad_dtypes:
        raise GradcheckError('Gradient has dtype mismatch')
    if not sizes_ok:
        raise GradcheckError('Analytical gradient has incorrect size')
    if not reentrant:
        raise GradcheckError(f'Backward is not reentrant, i.e., running backward with same input and grad_output multiple times gives different values, although analytical gradient matches numerical gradient.The tolerance for nondeterminism was {nondet_tol}.' + FAILED_NONDET_MSG)
    return jacobians1