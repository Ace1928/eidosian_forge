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
def _run_slow_mode_and_get_error(func, tupled_inputs, outputs, input_idx, output_idx, rtol, atol, is_forward_ad):
    slow_numerical = _get_numerical_jacobian(func, tupled_inputs, outputs, is_forward_ad=is_forward_ad)[input_idx][output_idx]
    if is_forward_ad:

        def new_fn(inp):
            new_inputs = list(tupled_inputs)
            new_inputs[input_idx] = inp
            return _as_tuple(func(*new_inputs))[output_idx]
        slow_analytical = _get_analytical_jacobian_forward_ad(new_fn, (tupled_inputs[input_idx],), (outputs[output_idx],))[0][0]
    else:
        slow_analytical = _get_analytical_jacobian(tupled_inputs, outputs, input_idx, output_idx)
    slow_max_diff = (slow_numerical - slow_analytical).abs().max()
    slow_allclose = torch.allclose(slow_analytical, slow_numerical, rtol, atol)
    msg = f'\nThe above quantities relating the numerical and analytical jacobians are computed \nin fast mode. See: https://github.com/pytorch/pytorch/issues/53876 for more background \nabout fast mode. Below, we recompute numerical and analytical jacobians in slow mode:\n\nNumerical:\n {slow_numerical}\nAnalytical:\n{slow_analytical}\n\nThe max per-element difference (slow mode) is: {slow_max_diff}.\n'
    if slow_allclose:
        msg += FAST_FAIL_SLOW_OK_MSG
    return msg