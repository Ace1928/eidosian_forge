from functools import partial
import torch
from .binary import (
from .core import is_masked_tensor, MaskedTensor, _get_data, _masks_match, _maybe_get_mask
from .passthrough import (
from .reductions import (
from .unary import (
@register_function_func(NATIVE_REDUCE_FNS + TORCH_REDUCE_FNS + TENSOR_REDUCE_FNS)
def _general_function_reductions(func, *args, **kwargs):
    return _apply_reduction(func, *args, **kwargs)