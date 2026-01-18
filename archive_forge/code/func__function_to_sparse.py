from functools import partial
import torch
from .binary import (
from .core import is_masked_tensor, MaskedTensor, _get_data, _masks_match, _maybe_get_mask
from .passthrough import (
from .reductions import (
from .unary import (
@register_function_func([torch.Tensor.to_sparse])
def _function_to_sparse(func, *args, **kwargs):
    return _MaskedToSparse.apply(args[0])