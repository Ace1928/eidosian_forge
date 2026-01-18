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
def _get_analytical_vJu_backward_mode(inputs, outputs, nondet_tol, check_grad_dtypes, all_v, all_u):
    reduced_jacobians: List[List[torch.Tensor]] = []
    for output, v in zip(outputs, all_v):
        all_vJ = _check_analytical_jacobian_attributes(inputs, output, nondet_tol, check_grad_dtypes, fast_mode=True, v=v)
        jacobian_scalars: List[torch.Tensor] = []
        for vJ, u in zip(all_vJ, all_u):
            vJ = vJ.T.squeeze(0)
            if vJ.is_complex():
                tv = torch.view_as_real(vJ.resolve_conj())
                tr = tv.select(-1, 0)
                ti = tv.select(-1, 1)
                jacobian_scalars.append(tr.dot(u[0]) + 1j * ti.dot(u[1]))
            else:
                jacobian_scalars.append(vJ.dot(u))
        reduced_jacobians.append(jacobian_scalars)
    return reduced_jacobians