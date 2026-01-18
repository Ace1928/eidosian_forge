import logging
import traceback
import warnings
import weakref
from enum import auto, Enum
from functools import partial
from typing import (
import torch
import torch.distributed as dist
import torch.distributed.fsdp._flat_param as flat_param_file
import torch.nn as nn
from torch.distributed._composable_state import _get_module_state, _State
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions
from torch.distributed.utils import _apply_to_tensors
from torch.utils._mode_utils import no_dispatch
from .api import (
def _named_parameters_with_duplicates(module: nn.Module, **kwargs: Any) -> List[Tuple[str, nn.Parameter]]:
    """
    This API is required as some modules overwrite `named_parameters()` but do not support
    `remove_duplicate`.
    """
    assert 'remove_duplicate' not in kwargs, '_named_parameters_with_duplicates cannot be used with `remove_duplicate` argument.'
    kwargs['remove_duplicate'] = False
    try:
        ret = list(module.named_parameters(**kwargs))
    except AssertionError as e:
        kwargs.pop('remove_duplicate')
        ret = list(module.named_parameters(**kwargs))
    return ret