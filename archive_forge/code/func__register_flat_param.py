import contextlib
import warnings
from typing import cast, Generator
import torch
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._runtime_utils import (
from torch.distributed.utils import _p_assert
from ._flat_param import FlatParamHandle
def _register_flat_param(state: _FSDPState, module: nn.Module) -> None:
    """
    Registers the flattened parameter to the wrapped module, making it
    visible to ``nn.Module`` methods.

    We do not use :meth:`nn.Module.register_parameter` because we want
    ``FLAT_PARAM`` to always be an attribute but dynamically change whether
    it is visible to ``nn.Module`` methods.
    """
    handle = _module_handle(state, module)
    if _has_fsdp_params(state, module):
        cast(nn.Module, module.module)._parameters[FLAT_PARAM] = handle.flat_param