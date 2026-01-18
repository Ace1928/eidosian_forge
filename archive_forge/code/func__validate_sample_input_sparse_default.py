import os
import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
def _validate_sample_input_sparse_default(op_info, sample, check_validate=False):
    if op_info.name == 'to_sparse':
        if sample.input.layout in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc} and len(sample.args) == 1 and isinstance(sample.args[0], int) and (sample.args[0] != 2):
            sample = ErrorInput(sample, error_regex='sparse dim argument must be 2 for sparse_compressed_to_sparse')
    if check_validate:
        _check_validate(op_info, sample)
    return sample