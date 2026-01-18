import torch
from .core import _map_mt_args_kwargs, _masks_match, _tensors_match, _wrap_result, is_masked_tensor
def _get_at_least_one_mask(a, b):
    if not is_masked_tensor(a) and (not is_masked_tensor(b)):
        raise TypeError('At least one of `a` and `b` must be a MaskedTensor')
    if not _masks_match(a, b):
        raise ValueError('a and b must have matching masks')
    if is_masked_tensor(a):
        return a.get_mask()
    return b.get_mask()