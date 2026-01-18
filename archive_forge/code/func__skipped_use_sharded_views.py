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
@property
def _skipped_use_sharded_views(self) -> bool:
    """
        This property is used for sharding strategies that do not free after
        forward with ``use_orig_params=True``. This returns if this handle is
        currently in a state where it has skipped using sharded views, in which
        case it can restore view invariants via ``_use_sharded_views()``.
        """
    return self._unsharded_flat_param_for_skipped_views is not None