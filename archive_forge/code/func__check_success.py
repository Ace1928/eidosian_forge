import os
import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
def _check_success(sample):
    try:
        op_info(sample.input, *sample.args, **sample.kwargs)
    except Exception as msg:
        raise AssertionError(f'{op_info.name} on sample={sample!r} expected to succeed , got {type(msg).__name__}: {msg}')