import os
import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
def _validate_sample_input_sparse_like_fns(op_info, sample, check_validate=False):
    if sample.input.layout in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}:
        if sample.kwargs.get('device', sample.input.device) != sample.input.device:
            return ErrorInput(sample, error_regex='device of (ccol|crow)_indices \\(=(cpu|cuda.*)\\) must match device of values \\(=(cuda.*|cpu)\\)')
        if sample.kwargs.get('layout', sample.input.layout) != sample.input.layout:
            return ErrorInput(sample, error_regex='empty_like with different sparse layout is not supported \\(self is Sparse(Csc|Csr|Bsc|Bsr) but you requested Sparse(Csr|Csc|Bsr|Bsc)\\)')
    if sample.input.layout is torch.sparse_coo:
        return ErrorInput(sample, error_regex="Could not run 'aten::normal_' with arguments from the 'Sparse(CPU|CUDA)' backend.")
    if check_validate:
        _check_validate(op_info, sample)
    return sample