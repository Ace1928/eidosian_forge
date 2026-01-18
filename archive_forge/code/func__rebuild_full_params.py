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
def _rebuild_full_params(self, force_full_precision: bool=False) -> Optional[List[Tuple[torch.Tensor, bool]]]:
    """
        Gather all shards of params.

        Note, this is idempotent if full params are already gathered. Callers
        assume the idempotency. So please keep it that way.

        Args:
            force_full_precision (bool, Optional): by default params will be gathered
                in ``compute_dtype`` (e.g., FP16), unless *force_full_precision* is
                ``True``, in which case they will be gathered in full precision
                (e.g., FP32), possibly in fresh storage. The parameter that's being
                rebuilt will end up in full precision as well.

        Returns:
            A list of tuples, where the first element is the full-sized param
            and the second element is a bool indicating if it's safe for the
            caller to free the full-sized param. This will be ``None`` if
            ``force_full_precision=False`` and the full params are already gathered.
        """
    output_tensors: List[Tuple[torch.Tensor, bool]] = []

    def update_p_data(custom_output_tensor: Optional[torch.Tensor]=None) -> None:
        """
            Helper function to update p.data pointer.

            Args:
                custom_output_tensor (torch.Tensor, Optional): if not None, this
                tensor contains the data we just gathered.
            """
        if custom_output_tensor is not None:
            assert p._is_sharded
            p.data = custom_output_tensor
            output_tensors.append((p.data, True))
        elif not p._is_sharded:
            if (self.mixed_precision or self.move_params_to_cpu) and (not force_full_precision):
                assert p._fp16_shard is not None
                p.data = p._fp16_shard
                output_tensors.append((p.data, True))
            else:
                output_tensors.append((p.data, False))
        else:
            p.data = p._full_param_padded
            output_tensors.append((p.data, True))
        p.data = p.data[:p._orig_size.numel()].view(p._orig_size)
    if self._has_shared_params:
        self.has_full_params = not any((p._full_param_padded.storage().size() == 0 for p in self.params))
    if self.has_full_params and (not force_full_precision):
        for p in self.params:
            update_p_data()
        return output_tensors
    self.has_full_params = True
    with torch.cuda.stream(self._streams['all_gather']):
        if (self.mixed_precision or self.move_params_to_cpu) and (not force_full_precision):
            self._cast_fp32_param_shards_to_fp16()
        if self.move_params_to_cpu:
            if force_full_precision:
                if self.params[0].dtype == self.compute_dtype:
                    self._cast_fp32_param_shards_to_fp16()
                else:
                    for p in self.params:
                        p.data = p.data.to(self.compute_device)
        for p in self.params:
            if not p._is_sharded:
                update_p_data()
            else:
                if p.data.shape == p._orig_size and p._orig_size != (1,):
                    assert p.data.storage().data_ptr() == p._full_param_padded.storage().data_ptr(), f'p.data {p.data.storage().data_ptr()} p._fp32_shard {p._fp32_shard.storage().data_ptr()} p._fp16_shard {(p._fp16_shard.storage().data_ptr() if p._fp16_shard is not None else None)} p._full_params_padded {p._full_param_padded.storage().data_ptr()} '
                    continue
                p_data = p.data.to(p._full_param_padded.device, non_blocking=True)
                full_p_size = p._full_param_padded.size()
                assert full_p_size.numel() % self.world_size == 0
                if self.mixed_precision and force_full_precision:
                    output_tensor = p_data.new_zeros(full_p_size)
                else:
                    if p._full_param_padded.storage().size() != full_p_size.numel():
                        alloc_storage_(p._full_param_padded, size=full_p_size)
                    output_tensor = p._full_param_padded
                if hasattr(dist, '_all_gather_base') and enable_nccl_base_collectives:
                    dist._all_gather_base(output_tensor, p_data, group=self.process_group)
                else:
                    chunks = list(output_tensor.chunk(self.world_size))
                    dist.all_gather(chunks, p_data, group=self.process_group)
                update_p_data(output_tensor)
                if (self.mixed_precision or self.move_params_to_cpu) and (not force_full_precision):
                    self._free_fp16_param_shard([p])
                if self.move_params_to_cpu and self.params[0].dtype == self.compute_dtype:
                    self._free_fp16_param_shard([p])
    torch.cuda.current_stream().wait_stream(self._streams['all_gather'])
    return output_tensors