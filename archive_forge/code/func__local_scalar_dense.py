from functools import partial
import torch
from .binary import (
from .core import is_masked_tensor, MaskedTensor, _get_data, _masks_match, _maybe_get_mask
from .passthrough import (
from .reductions import (
from .unary import (
@register_dispatch_func([torch.ops.aten._local_scalar_dense])
def _local_scalar_dense(func, *args, **kwargs):
    if not _maybe_get_mask(args[0]):
        raise ValueError(f'__torch_dispatch__, {func}: expected a mask tensor')
    return torch.ops.aten._local_scalar_dense(_get_data(args[0]))