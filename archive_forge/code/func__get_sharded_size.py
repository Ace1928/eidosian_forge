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
@staticmethod
def _get_sharded_size(tensor: Tensor, rank: int, world_size: int) -> torch.Size:
    """
        Returns the shape of ``tensor`` after sharding including padding. This
        requires ``tensor`` to have 1D shape and ensures that the returned
        shape is 1D.
        """
    assert len(tensor.shape) == 1, f'{tensor.shape}'
    unpadded_sharded_tensor, numel_to_pad = FlatParamHandle._get_unpadded_shard(tensor, rank, world_size)
    unpadded_sharded_size = unpadded_sharded_tensor.size()
    assert len(unpadded_sharded_size) == 1, f'{unpadded_sharded_size}'
    return torch.Size([unpadded_sharded_size[0] + numel_to_pad])