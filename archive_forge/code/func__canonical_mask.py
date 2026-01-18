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
def _canonical_mask(mask: Optional[Tensor], mask_name: str, other_type: Optional[DType], other_name: str, target_type: DType, check_other: bool=True) -> Optional[Tensor]:
    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and (not _mask_is_float):
            raise AssertionError(f'only bool and floating types of {mask_name} are supported')
        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                warnings.warn(f'Support for mismatched {mask_name} and {other_name} is deprecated. Use same type for both instead.')
        if not _mask_is_float:
            mask = torch.zeros_like(mask, dtype=target_type).masked_fill_(mask, float('-inf'))
    return mask