import os
import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
def _check_validate(op_info, sample):

    def _check_fail(sample):
        try:
            op_info(sample.sample_input.input, *sample.sample_input.args, **sample.sample_input.kwargs)
        except sample.error_type:
            pass
        except Exception as msg:
            raise AssertionError(f'{op_info.name} on sample.sample_input={sample.sample_input!r} expected exception {sample.error_type}: {sample.error_regex}, got {type(msg).__name__}: {msg}')
        else:
            raise AssertionError(f'{op_info.name} on sample.sample_input={sample.sample_input!r} expected exception {sample.error_type}: {sample.error_regex}, got none.')

    def _check_success(sample):
        try:
            op_info(sample.input, *sample.args, **sample.kwargs)
        except Exception as msg:
            raise AssertionError(f'{op_info.name} on sample={sample!r} expected to succeed , got {type(msg).__name__}: {msg}')
    if isinstance(sample, ErrorInput):
        _check_fail(sample)
    else:
        _check_success(sample)