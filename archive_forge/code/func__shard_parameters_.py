import contextlib
import copy
from enum import Enum, auto
import functools
import logging
from math import inf
import os
import time
import traceback
import typing
from typing import (
import torch
from torch.autograd import Variable
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from fairscale.internal.containers import apply_to_tensors
from fairscale.internal.parallel import (
from fairscale.internal.params import calc_grad_norm, recursive_copy_to_device
from fairscale.internal.reduce_scatter_bucketer import ReduceScatterBucketer
from fairscale.internal.state_dict import replace_by_prefix_
from fairscale.nn.misc import FlattenParamsWrapper, _enable_pre_load_state_dict_hook
from fairscale.nn.wrap import auto_wrap, config_auto_wrap_policy, enable_wrap
from . import fsdp_optim_utils as ou
@torch.no_grad()
def _shard_parameters_(self) -> None:
    """
        At initialization we wrap a module with full parameters and shard the
        parameters in-place. Sharding is implemented by viewing each parameter
        as a 1D Tensor and retaining only a single slice, where the slice size
        is determined by the number of data parallel workers.

        Wrapping modules with many small parameters (or with a very large data
        parallel world size) will result in many small parameter shards and slow
        performance. In this case it's better to set *``flatten_parameters``* to
        ``True``, so that all of the small parameters in the module are combined
        into a single contiguous Tensor and sharded once.

        After this initial sharding is complete, the user can initialize a
        ``torch.optim.Optimizer`` in the usual way, i.e.::

        .. code-block:: python

            optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)

        The optimizer will see only a single slice of parameters and will thus
        allocate less memory for optimizer state, avoiding redundancy across
        data parallel workers.
        """
    self.numel_padded_per_param = []
    for p in self.params:
        assert not hasattr(p, '_is_sharded')
        assert p.is_floating_point()
        if self.mixed_precision:
            assert p.dtype == torch.float32
        p._is_sharded = self.world_size > 1
        p._orig_size = p.data.size()
        if not p._is_sharded:
            p._is_sharded = False
            self.numel_padded_per_param.append(0)
            continue
        p._is_sharded = True
        orig_data = p.data
        p.data, num_padded = self._get_shard(p.data)
        self.numel_padded_per_param.append(num_padded)
        free_storage_(orig_data)
    assert len(self.numel_padded_per_param) == len(self.params)