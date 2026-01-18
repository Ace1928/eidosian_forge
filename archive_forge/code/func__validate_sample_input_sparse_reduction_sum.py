import os
import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
def _validate_sample_input_sparse_reduction_sum(sample, check_validate=False):
    t_inp, t_args, t_kwargs = (sample.input, sample.args, sample.kwargs)
    dim = t_kwargs.get('dim')
    keepdim = t_kwargs.get('keepdim')
    layout = t_inp.layout
    if isinstance(dim, (int, list, tuple)):
        if layout in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}:
            if layout in {torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}:
                return ErrorInput(sample, error_regex='Currently the only compressed sparse format supported for sum.dim_IntList is CSR, but got layout')
            if layout in {torch.sparse_csr, torch.sparse_csc} and (not keepdim):
                return ErrorInput(sample, error_regex='reduction operations on CSR tensors with keepdim=False is unsupported')
            if t_inp.dim() != 2:
                return ErrorInput(sample, error_regex='input_dim == 2 INTERNAL ASSERT')
            if layout == torch.sparse_csr:
                if t_inp.dtype == torch.bool:
                    return ErrorInput(sample, error_regex="_sparse_csr_sum_cpu not implemented for 'Bool'")
                if t_inp.dtype == torch.complex32:
                    return ErrorInput(sample, error_regex="_sparse_csr_sum_cuda not implemented for 'ComplexHalf'")
    return sample