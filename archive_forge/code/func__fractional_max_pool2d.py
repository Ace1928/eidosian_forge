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
def _fractional_max_pool2d(input: Tensor, kernel_size: BroadcastingList2[int], output_size: Optional[BroadcastingList2[int]]=None, output_ratio: Optional[BroadcastingList2[float]]=None, return_indices: bool=False, _random_samples: Optional[Tensor]=None) -> Tensor:
    if has_torch_function_variadic(input, _random_samples):
        return handle_torch_function(fractional_max_pool2d, (input, _random_samples), input, kernel_size, output_size=output_size, output_ratio=output_ratio, return_indices=return_indices, _random_samples=_random_samples)
    return fractional_max_pool2d_with_indices(input, kernel_size, output_size, output_ratio, return_indices, _random_samples)[0]