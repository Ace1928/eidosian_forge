import os
import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
def _validate_sample_input_sparse_reduction(op_info, sample, check_validate=False):
    """Return the specified sample when it is valid and supported by the
    operation. Otherwise, return the sample as ErrorInput instance.

    When check_validate is True, the result is validated against
    calling the op on the sample.
    """
    UNSPECIFIED = object()
    if op_info.name == 'sum':
        sample = _validate_sample_input_sparse_reduction_sum(sample)
    if op_info.name in {'masked.sum'}:
        mask = sample.kwargs.get('mask', UNSPECIFIED)
        if mask not in {None, UNSPECIFIED} and mask.ndim > 2 and (mask.layout is torch.strided) and (mask == 0).any():
            sample = ErrorInput(sample, error_regex='Expect the same number of specified elements per batch.')
        elif not sample.kwargs.get('keepdim'):
            sample = ErrorInput(sample, error_type=(AssertionError, RuntimeError), error_regex='reduction operations on (CSR|CSC) tensors with keepdim=False is unsupported')
        elif mask is UNSPECIFIED:
            sample = ErrorInput(sample, error_type=ValueError, error_regex='masked (.*) expects explicit mask for sparse_csr tensor input')
        elif sample.input.ndim > 2:
            sample = ErrorInput(sample, error_regex='crow_indices is supposed to be a vector, but got 3 dimensional tensor.')
    if op_info.name in {'masked.amax', 'masked.amin', 'masked.mean', 'masked.prod'}:
        t_inp = sample.input
        batch_dim = t_inp.dim() - t_inp.dense_dim() - t_inp.sparse_dim()
        mask = sample.kwargs.get('mask')
        if mask is not None and mask.ndim > 2 and (mask.layout is torch.strided) and (mask == 0).any():
            sample = ErrorInput(sample, error_regex='Expect the same number of specified elements per batch.')
        elif mask is None:
            sample = ErrorInput(sample, error_type=ValueError, error_regex='masked (.*) expects explicit mask for sparse_csr tensor input')
        elif mask.layout is sample.input.layout and mask.ndim > 2 and (op_info.name == 'masked.mean'):
            sample = ErrorInput(sample, error_type=TypeError, error_regex='where[(][)] received an invalid combination of arguments - got [(]Tensor, Tensor, NoneType[)]')
        elif not sample.kwargs.get('keepdim'):
            sample = ErrorInput(sample, error_type=(AssertionError, RuntimeError), error_regex='reduction operations on (CSR|CSC) tensors with keepdim=False is unsupported')
        elif sample.input.ndim > 2 and sample.kwargs.get('dim') not in {0, 1} and (mask.ndim > 2) and (mask.layout is not torch.strided):
            if sample.kwargs.get('dim') == (0, -1):
                sample = ErrorInput(sample, error_regex='tensor dimensionality must be sum of batch, base, and dense dimensionalities')
            elif op_info.name == 'masked.prod':
                sample = ErrorInput(sample, error_regex='input_dim == 2 INTERNAL ASSERT FAILED at')
            else:
                sample = ErrorInput(sample, error_type=AssertionError, error_regex='Sparse CSR tensors are 2D and only support reduction along dim 0 or 1.')
        elif sample.input.ndim > 2:
            sample = ErrorInput(sample, error_regex='crow_indices is supposed to be a vector, but got 3 dimensional tensor.')
        elif mask.layout is t_inp.layout and mask._nnz() != t_inp._nnz() and (t_inp.dense_dim() > 0):
            sample = ErrorInput(sample, error_regex='Index tensor must have the same number of dimensions as src tensor')
    if check_validate:
        _check_validate(op_info, sample)
    return sample