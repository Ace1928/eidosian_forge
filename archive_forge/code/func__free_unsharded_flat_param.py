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
def _free_unsharded_flat_param(self):
    """
        Frees the padded unsharded flat parameter. The tensor to free depends
        on the calling context since the unshard may have forced full
        precision, in which case a different tensor is used.
        """
    self._check_sharded_strategy()
    unsharded_flat_param = self._get_padded_unsharded_flat_param()
    self._check_storage_allocated(unsharded_flat_param)
    self._check_on_compute_device(unsharded_flat_param)
    _no_dispatch_record_stream(unsharded_flat_param, self._device_handle.current_stream())
    _free_storage(unsharded_flat_param)