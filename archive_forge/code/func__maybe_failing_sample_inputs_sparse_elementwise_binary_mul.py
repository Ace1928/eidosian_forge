import os
import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
@_apply_requires_grad_to_samples
def _maybe_failing_sample_inputs_sparse_elementwise_binary_mul(op_info, device, dtype, requires_grad, layout, **kwargs):
    """Generator of samples that are known to fail or that were failing in past."""
    blocksize = (1, 1) if layout in {torch.sparse_bsr, torch.sparse_bsc} else None
    regular = torch.tensor([[1, 2], [3, 4]], device=device, dtype=dtype).to_sparse(layout=layout, dense_dim=0, blocksize=blocksize)
    batch = torch.tensor([[[1, 2], [3, 4]], [[4, 5], [6, 7]]], device=device, dtype=dtype).to_sparse(layout=layout, dense_dim=0, blocksize=blocksize)
    hybrid = torch.tensor([[[1], [2]], [[3], [4]]], device=device, dtype=dtype).to_sparse(layout=layout, dense_dim=1, blocksize=blocksize)
    if layout is torch.sparse_csr:
        yield SampleInput(batch, args=(batch,))
        yield SampleInput(torch.zeros_like(hybrid).requires_grad_(requires_grad), args=(torch.zeros_like(hybrid).requires_grad_(requires_grad),))
        if dtype is torch.complex32:
            yield SampleInput(regular, args=(regular,))
        if dtype is torch.bool and regular.is_cpu:
            yield SampleInput(regular, args=(regular,))
    if layout is torch.sparse_csc:
        yield SampleInput(regular, args=(regular,))
    if layout is torch.sparse_bsr:
        yield SampleInput(regular, args=(regular,))
    if layout is torch.sparse_bsc:
        yield SampleInput(regular, args=(regular,))
    if layout is torch.sparse_coo:
        if dtype is torch.complex32:
            yield SampleInput(regular, args=(regular,))
        if dtype is torch.bool and regular.is_cpu:
            yield SampleInput(regular, args=(regular,))
        if dtype in {torch.bool, torch.float16} and regular.is_cpu:
            yield SampleInput(hybrid, args=(hybrid,))