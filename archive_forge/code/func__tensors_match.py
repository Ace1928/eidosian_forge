import warnings
import torch
from torch.overrides import get_default_nowrap_functions
def _tensors_match(a, b, exact=True, rtol=1e-05, atol=1e-08):
    if is_masked_tensor(a) or is_masked_tensor(b):
        raise ValueError('Neither `a` nor `b` can be a MaskedTensor.')
    if a.layout != b.layout:
        raise ValueError(f'`a` and `b` must have the same layout. Got {a.layout} and {b.layout}')
    if a.dtype != b.dtype:
        b = b.type(a.dtype)
    if a.layout == b.layout == torch.sparse_coo:
        return _tensors_match(a.values(), b.values(), exact) and _tensors_match(a.indices(), b.indices(), exact)
    elif a.layout == b.layout == torch.sparse_csr:
        return _tensors_match(a.crow_indices(), b.crow_indices(), exact) and _tensors_match(a.col_indices(), b.col_indices(), exact) and _tensors_match(a.values(), b.values(), exact)
    if exact:
        return a.dim() == b.dim() and torch.eq(a, b).all().item()
    return a.dim() == b.dim() and torch.allclose(a, b, rtol=rtol, atol=atol)