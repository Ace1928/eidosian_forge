import contextlib
import functools
import logging
import os
import warnings
from enum import auto, Enum
from itertools import accumulate, chain
from typing import (
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.fsdp._common_utils import (
from torch.distributed.utils import _alloc_storage, _free_storage, _p_assert
from torch.nn.parameter import _ParameterMeta  # type: ignore[attr-defined]
from ._fsdp_extensions import _ext_post_unflatten_transform, _ext_pre_flatten_transform
class SharedParamInfo(NamedTuple):
    """
    Additional information for a shared parameter.

    For each shared parameter, we designate one module and its parameter
    variable to be the primary owner, determined as the first one encountered
    in the parameter walk. These are prefixed with "prim". The primary module
    and parameter do not have their own :class:`SharedParamInfo` instance.
    """
    param_name: str
    module: nn.Module
    module_name: str
    prim_param_name: str
    prim_module: nn.Module
    prim_module_name: str