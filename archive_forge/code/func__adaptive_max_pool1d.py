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
def _adaptive_max_pool1d(input: Tensor, output_size: BroadcastingList1[int], return_indices: bool=False) -> Tensor:
    if has_torch_function_unary(input):
        return handle_torch_function(adaptive_max_pool1d, (input,), input, output_size, return_indices=return_indices)
    return adaptive_max_pool1d_with_indices(input, output_size)[0]