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
@no_type_check
def _get_unflat_views_unaligned(self, tensor: Optional[torch.Tensor]=None) -> Iterator[Tensor]:
    """
        Returns unflattened ``Tensor`` views into ``tensor`` if it is not
        ``None`` or ``flat_param`` otherwise, where the unflattening is based
        on ``flat_param`` 's metadata.

        Examples for ``tensor`` include ``flat_param.grad`` or unsharded
        tensor optimizer state.
        """
    flat_param = self.flat_param
    if tensor is None:
        tensor = flat_param
    views = (_ext_post_unflatten_transform(subtensor.view(shape), param_extension) for subtensor, shape, param_extension in zip(torch.split(tensor, flat_param._numels, dim=0), flat_param._shapes, flat_param._param_extensions))
    return views