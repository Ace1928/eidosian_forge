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
def _init_param_attributes(self, p: Parameter) -> None:
    """
        We manage several attributes on each Parameter instance. The first two
        are set by :func:`_shard_parameters_`:

            ``_is_sharded``: ``True`` if the Parameter is sharded or ``False``
                if the Parameter is intentionally not sharded (in which case we
                will all-reduce grads for this param).
            ``_orig_size``: the size of the original Parameter (before sharding)

        The remaining attributes are set here:
            ``_fp32_shard``: a single shard of the parameters in full precision
                (typically FP32, but this is dependent on the dtype of the model
                as it's passed in by the user). This can be on CPU or GPU
                depending on the value of *``move_params_to_cpu``*.
            ``_fp16_shard``: This will be a single shard of the parameters in FP16, used for all-gather.
                This can be in FP16 or FP32 depending on the value of *``compute_dtype``* and
                if params are offloaded to CPU.
            ``_full_param_padded``: the full weight (padded to be evenly
                divisible by ``world_size``), used for computation in the
                forward and backward pass. This will be resized in place and
                only materialized (via all-gather) as needed.
        """
    assert hasattr(p, '_is_sharded') and hasattr(p, '_orig_size')
    if hasattr(p, '_fp32_shard'):
        return
    p._fp32_shard = p.data
    if self.mixed_precision:
        assert p._fp32_shard.dtype == torch.float32, self
    if self.move_params_to_cpu:
        assert p._fp32_shard.device == torch.device('cpu'), self
        p._fp32_shard = p._fp32_shard.pin_memory()
        p.data = p._fp32_shard
    if self.move_params_to_cpu or self.mixed_precision:
        p._fp16_shard = torch.zeros_like(p._fp32_shard, device=self.compute_device, dtype=self.compute_dtype)
        free_storage_(p._fp16_shard)
    if self.mixed_precision:
        assert p._fp32_shard.dtype == torch.float32
    if not self.mixed_precision and (not self.move_params_to_cpu):
        p._fp16_shard = None
    if p._is_sharded:
        p._full_param_padded = torch.zeros(p.data.numel() * self.world_size, device=self.compute_device, dtype=self.compute_dtype)
        free_storage_(p._full_param_padded)
    if self.move_grads_to_cpu and self.training:
        p._cpu_grad = torch.zeros_like(p.data, device='cpu').pin_memory()