from functools import partial
import torch
from .binary import (
from .core import is_masked_tensor, MaskedTensor, _get_data, _masks_match, _maybe_get_mask
from .passthrough import (
from .reductions import (
from .unary import (
@register_dispatch_func([torch.ops.aten._to_sparse])
def _to_sparse(func, *args, **kwargs):
    _check_args_kwargs_length(args, kwargs, f'__torch_dispatch__, {func}', len_args=1, len_kwargs=0)
    if not torch.is_tensor(args[0]):
        raise TypeError('__torch_dispatch__, {func}: expected args[0] to be a tensor')
    mt = args[0]
    if not is_masked_tensor(mt):
        mt = MaskedTensor(mt, torch.ones_like(mt, dtype=torch.bool))
    if mt.is_sparse_coo():
        return mt
    new_mask = func(_maybe_get_mask(args[0])).coalesce()
    new_data = _get_data(args[0]).sparse_mask(new_mask)
    return MaskedTensor(new_data, new_mask)