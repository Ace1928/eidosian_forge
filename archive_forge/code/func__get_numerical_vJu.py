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
def _get_numerical_vJu(fn, inputs, inp_indices, func_out, all_u, all_v, eps, is_forward_ad):
    reduced_jacobians: List[List[torch.Tensor]] = []
    for i, (inp_idx, u) in enumerate(zip(inp_indices, all_u)):
        all_Ju = _get_numerical_jvp_wrt_specific_input(fn, inp_idx, inputs, u, eps, is_forward_ad)
        filtered_Ju = []
        func_out = _as_tuple(func_out)
        assert len(all_Ju) == len(func_out)
        for Ju, output in zip(all_Ju, func_out):
            if _is_float_or_complex_tensor(output):
                filtered_Ju.append(Ju)
            else:
                pass
        if all_v is not None:
            jacobian_scalars: List[torch.Tensor] = []
            for v, Ju in zip(all_v, filtered_Ju):
                jacobian_scalars.append(_dot_with_type_promotion(v, Ju))
            reduced_jacobians.append(jacobian_scalars)
        else:
            reduced_jacobians.append(filtered_Ju)
    return reduced_jacobians