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
def _get_shard(tensor: Tensor, rank: int, world_size: int) -> Tuple[Tensor, int]:
    """
        Returns the shard of ``tensor`` with padding for the given ``rank`` and
        ``world_size`` and the numel padded for that shard.

        This method allocates new memory (via :meth:`clone`) since the
        unsharded ``tensor`` may be deallocated after this method returns.
        """
    chunk, numel_to_pad = FlatParamHandle._get_unpadded_shard(tensor, rank, world_size)
    shard = chunk.clone()
    if numel_to_pad > 0:
        shard = F.pad(shard, [0, numel_to_pad])
    return (shard, numel_to_pad)