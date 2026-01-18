from functools import partial
import torch
from .binary import (
from .core import is_masked_tensor, MaskedTensor, _get_data, _masks_match, _maybe_get_mask
from .passthrough import (
from .reductions import (
from .unary import (

    Used for registering a new __torch_dispatch__ function to MaskedTensor
    Called via _MASKEDTENSOR_DISPATCH_TABLE[func](*args, **kwargs)

    The code to register a new function looks like:

    @register_dispatch_func(list_of_ops)
    def foo(func, *args, **kwargs):
        <implementation>
    