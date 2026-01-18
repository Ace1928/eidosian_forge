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
def _finalize_parameters(fsdp_module: FullyShardedDataParallel) -> None:
    """Helper used below on all fsdp modules."""
    if not fsdp_module._is_root and self._require_backward_grad_sync:
        fsdp_module._free_full_params()
        fsdp_module._use_fp32_param_shard()
    for p in fsdp_module.params:
        if not p.requires_grad:
            continue
        if not self._require_backward_grad_sync:
            continue
        if hasattr(p, '_cpu_grad'):
            p_assert(p.device == torch.device('cpu'), f'WFPB: incorrect cpu_grad device {p.device}')
            p.grad = p._cpu_grad
        elif hasattr(p, '_saved_grad_shard'):
            p_assert(p.device == p._saved_grad_shard.device, f'WFPB: incorrect saved_grad_shard device p.device={p.device} vs p._saved_grad_shard.device={p._saved_grad_shard.device}')
            p_assert(p.shape == p._saved_grad_shard.shape, f'WFPB: incorrect saved_grad_shard shape p.shape={p.shape} vs p._saved_grad_shard.shape={p._saved_grad_shard.shape}')
            p.grad = p._saved_grad_shard
        if hasattr(p, '_saved_grad_shard'):
            delattr(p, '_saved_grad_shard')