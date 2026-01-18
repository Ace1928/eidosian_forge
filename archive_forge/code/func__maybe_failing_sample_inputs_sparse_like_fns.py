import os
import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
def _maybe_failing_sample_inputs_sparse_like_fns(op_info, device, dtype, requires_grad, layout, **kwargs):
    if torch.cuda.is_available() and layout is not torch.sparse_coo:
        other_device = 'cuda' if torch.device(device).type == 'cpu' else 'cpu'
        if layout is torch.sparse_csr:
            other_layout = torch.sparse_csc
        elif layout is torch.sparse_csc:
            other_layout = torch.sparse_csr
        elif layout is torch.sparse_bsr:
            other_layout = torch.sparse_bsc
        elif layout is torch.sparse_bsc:
            other_layout = torch.sparse_bsr
        else:
            other_layout = torch.strided
        blocksize = (1, 1) if layout in {torch.sparse_bsr, torch.sparse_bsc} else None
        yield SampleInput(torch.tensor([[0, 1], [2, 3]], dtype=dtype, device=device).to_sparse(layout=layout, blocksize=blocksize), kwargs=dict(device=other_device))
        yield SampleInput(torch.tensor([[0, 1], [2, 3]], dtype=dtype, device=device).to_sparse(layout=layout, blocksize=blocksize), kwargs=dict(layout=other_layout))