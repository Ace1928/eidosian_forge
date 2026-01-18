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
def _check_no_differentiable_outputs(func, inputs, func_out, eps, *, is_forward_ad) -> bool:
    jacobians_all_inputs_outputs = _get_numerical_jacobian(func, inputs, func_out, eps=eps, is_forward_ad=is_forward_ad)
    for jacobians_all_outputs_and_fixed_input in jacobians_all_inputs_outputs:
        for jacobian in jacobians_all_outputs_and_fixed_input:
            if torch.ne(jacobian, 0).sum() > 0:
                raise GradcheckError('Numerical gradient for function expected to be zero')
    return True