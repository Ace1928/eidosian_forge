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
def _free_low_precision_sharded_param(self):
    """Frees the low precision sharded flat parameter."""
    self._check_low_precision_shard()
    _no_dispatch_record_stream(self.flat_param._mp_shard, self._device_handle.current_stream())
    _free_storage(self.flat_param._mp_shard)