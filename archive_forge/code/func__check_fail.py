import os
import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
def _check_fail(sample):
    try:
        op_info(sample.sample_input.input, *sample.sample_input.args, **sample.sample_input.kwargs)
    except sample.error_type:
        pass
    except Exception as msg:
        raise AssertionError(f'{op_info.name} on sample.sample_input={sample.sample_input!r} expected exception {sample.error_type}: {sample.error_regex}, got {type(msg).__name__}: {msg}')
    else:
        raise AssertionError(f'{op_info.name} on sample.sample_input={sample.sample_input!r} expected exception {sample.error_type}: {sample.error_regex}, got none.')