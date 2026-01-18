from typing import Callable, List, Optional, Tuple, Union
import math
import warnings
import importlib
import torch
from torch import _VF
from torch import sym_int as _sym_int
from torch._C import _infer_size, _add_docstr
from torch._torch_docs import reproducibility_notes, tf32_notes, sparse_support_notes
from typing import TYPE_CHECKING
from .._jit_internal import boolean_dispatch, _overload, BroadcastingList1, BroadcastingList2, BroadcastingList3
from ..overrides import (
from . import _reduction as _Reduction
from . import grad  # noqa: F401
from .modules import utils
from .modules.utils import _single, _pair, _triple, _list_with_default
def _unpool_output_size(input: Tensor, kernel_size: List[int], stride: List[int], padding: List[int], output_size: Optional[List[int]]) -> List[int]:
    input_size = input.size()
    default_size = torch.jit.annotate(List[int], [])
    for d in range(len(kernel_size)):
        default_size.append((input_size[-len(kernel_size) + d] - 1) * stride[d] + kernel_size[d] - 2 * padding[d])
    if output_size is None:
        ret = default_size
    else:
        if len(output_size) == len(kernel_size) + 2:
            output_size = output_size[2:]
        if len(output_size) != len(kernel_size):
            raise ValueError(f"output_size should be a sequence containing {len(kernel_size)} or {len(kernel_size) + 2} elements, but it has a length of '{len(output_size)}'")
        for d in range(len(kernel_size)):
            min_size = default_size[d] - stride[d]
            max_size = default_size[d] + stride[d]
            if not min_size < output_size[d] < max_size:
                raise ValueError(f'invalid output_size "{output_size}" (dim {d} must be between {min_size} and {max_size})')
        ret = output_size
    return ret