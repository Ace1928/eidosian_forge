import os
import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
def _validate_sample_input_sparse_elementwise_binary_operation(op_info, sample, check_validate=False):
    if op_info.name == 'mul':
        sample = _validate_sample_input_elementwise_binary_sparse_mul(sample)
    if check_validate:
        _check_validate(op_info, sample)
    return sample