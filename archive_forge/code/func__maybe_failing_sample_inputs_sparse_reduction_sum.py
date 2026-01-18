import os
import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
def _maybe_failing_sample_inputs_sparse_reduction_sum(op_info, device, dtype, requires_grad, layout, **kwargs):
    """Generator of samples that are known to fail or that were failing in past."""
    if layout in [torch.sparse_csr, torch.sparse_csc]:
        yield SampleInput(torch.tensor([[0, 1], [2, 3]], dtype=dtype).to_sparse(layout=layout).requires_grad_(requires_grad), kwargs=dict(dim=0, keepdim=True))
        yield SampleInput(torch.tensor([[[0, 1]], [[2, 3]]], dtype=dtype).to_sparse(layout=layout, dense_dim=1).requires_grad_(requires_grad), kwargs=dict(dim=0))
        yield SampleInput(torch.tensor([[0, 1], [2, 3]], dtype=dtype).to_sparse(layout=layout).requires_grad_(requires_grad), kwargs=dict(dim=(0,)))
        yield SampleInput(torch.tensor([[0, 1], [2, 3]], dtype=dtype).to_sparse(layout=layout).requires_grad_(requires_grad), kwargs=dict(dim=(0,), keepdim=True))
        yield SampleInput(torch.tensor([[[0, 1]], [[2, 3]]], dtype=dtype).to_sparse(layout=layout, dense_dim=1).requires_grad_(requires_grad), kwargs=dict(dim=(0,)))
        yield SampleInput(torch.tensor([[0, 1], [2, 3]], dtype=dtype).to_sparse(layout=layout).requires_grad_(requires_grad), kwargs=dict(dim=0))
    if layout in [torch.sparse_bsr, torch.sparse_bsc]:
        yield SampleInput(torch.tensor([[0, 1], [2, 3]], dtype=dtype).to_sparse(layout=layout, blocksize=(2, 2)).requires_grad_(requires_grad), kwargs=dict(dim=0, keepdim=True))
        yield SampleInput(torch.tensor([[[0, 1]], [[2, 3]]], dtype=dtype).to_sparse(layout=layout, dense_dim=1, blocksize=(1, 1)).requires_grad_(requires_grad), kwargs=dict(dim=0))
        yield SampleInput(torch.tensor([[0, 1], [2, 3]], dtype=dtype).to_sparse(layout=layout, blocksize=(1, 1)).requires_grad_(requires_grad), kwargs=dict(dim=(0,)))
        yield SampleInput(torch.tensor([[0, 1], [2, 3]], dtype=dtype).to_sparse(layout=layout, blocksize=(1, 1)).requires_grad_(requires_grad), kwargs=dict(dim=(0,), keepdim=True))
        yield SampleInput(torch.tensor([[[0, 1]], [[2, 3]]], dtype=dtype).to_sparse(layout=layout, blocksize=(1, 1), dense_dim=1).requires_grad_(requires_grad), kwargs=dict(dim=(0,)))
        yield SampleInput(torch.tensor([[0, 1], [2, 3]], dtype=dtype).to_sparse(layout=layout, blocksize=(1, 1)).requires_grad_(requires_grad), kwargs=dict(dim=0))