from functools import partial
import torch
from .binary import (
from .core import is_masked_tensor, MaskedTensor, _get_data, _masks_match, _maybe_get_mask
from .passthrough import (
from .reductions import (
from .unary import (
def _check_args_kwargs_length(args, kwargs, error_prefix, len_args=None, len_kwargs=None):
    if len_args is not None and len_args != len(args):
        raise ValueError(f'{error_prefix}: len(args) must be {len_args} but got {len(args)}')
    if len_kwargs is not None and len_kwargs != len(kwargs):
        raise ValueError(f'{error_prefix}: len(kwargs) must be {len_kwargs} but got {len(kwargs)}')