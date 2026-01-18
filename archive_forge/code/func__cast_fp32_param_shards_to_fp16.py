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
def _cast_fp32_param_shards_to_fp16(self, params: Optional[List[Parameter]]=None) -> None:
    """Cast FP32 param shard to FP16 for a list of params."""
    if params is None:
        params = self.params
    with torch.cuda.stream(self._streams['fp32_to_fp16']):
        for p in params:
            assert p._fp16_shard is not None
            alloc_storage_(p._fp16_shard, size=p._fp32_shard.size())
            p._fp16_shard.copy_(p._fp32_shard.to(p._fp16_shard.device, non_blocking=True))
            p.data = p._fp16_shard
    torch.cuda.current_stream().wait_stream(self._streams['fp32_to_fp16'])