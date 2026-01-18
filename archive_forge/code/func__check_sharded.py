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
def _check_sharded(self, tensor: Tensor):
    msg_prefix = 'Expects tensor to be sharded '
    _p_assert(tensor is not None, msg_prefix + 'but got `None`')
    sharded_size = self.flat_param._sharded_size
    _p_assert(tensor.size() == sharded_size, msg_prefix + f'with size {sharded_size} but got {tensor.size()}')