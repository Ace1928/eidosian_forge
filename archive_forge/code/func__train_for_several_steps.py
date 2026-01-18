import itertools
import os
import re
import sys
from abc import ABC, abstractmethod
from contextlib import nullcontext
from copy import deepcopy
from enum import auto, Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from unittest import mock
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import TrainingState
from torch.distributed.fsdp._init_utils import NO_RESHARD_AFTER_FORWARD_STRATEGIES
from torch.distributed.fsdp.fully_sharded_data_parallel import (
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import always_wrap_policy, ModuleWrapPolicy, wrap
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import (
from torch.testing._internal.common_utils import FILE_SCHEMA, get_cycles_per_ms
def _train_for_several_steps(self, model: nn.Module, num_steps: int, autocast: bool, lr: float=0.01, fsdp_cpu_offload: Optional[CPUOffload]=None, save_model: bool=False, mixed_precision: Optional[MixedPrecision]=None, enable_sharded_grad_scaler: bool=False, use_pure_fp16: bool=False, sharded_grad_scaler_kwargs: Optional[Dict[str, Any]]=None):
    cpu_offload_params = fsdp_cpu_offload and fsdp_cpu_offload.offload_params
    model_device = next(model.parameters()).device
    if sharded_grad_scaler_kwargs is None:
        sharded_grad_scaler_kwargs = {}
    sharded_grad_scaler = ShardedGradScaler(enabled=enable_sharded_grad_scaler, **sharded_grad_scaler_kwargs)
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for _ in range(num_steps):
        optim.zero_grad()
        with torch.cuda.amp.autocast(enabled=autocast):
            input = model.module.get_input(torch.device('cuda'))
            if use_pure_fp16 or (mixed_precision and (not isinstance(model, FSDP))):
                if isinstance(input, torch.Tensor):
                    input = input.half()
                else:
                    input = tuple((x.half() for x in input))
            output = model(*input)
            if cpu_offload_params and isinstance(model, FSDP) and (model.sharding_strategy not in NO_RESHARD_AFTER_FORWARD_STRATEGIES):
                for p in model.parameters():
                    self.assertEqual(p.device, torch.device('cpu'))
            loss = model.module.get_loss(input, output).to(model_device)
        loss = sharded_grad_scaler.scale(loss)
        if not mixed_precision and (not use_pure_fp16):
            assert loss.dtype == torch.float32, 'loss data type should be float32, as the original                     parameter data type is float32.'
        elif use_pure_fp16:
            self.assertEqual(loss.dtype, torch.float16)
        elif isinstance(model, FSDP):
            self.assertEqual(loss.dtype, mixed_precision.param_dtype)
        else:
            self.assertEqual(loss.dtype, torch.float32)
        model.module.run_backward(loss)
        if cpu_offload_params and isinstance(model, FSDP):
            for p in model.parameters():
                self.assertEqual(p.device, torch.device('cpu'))
        sharded_grad_scaler.step(optim)
        sharded_grad_scaler.update()
        if save_model:
            state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            _zero_model(model)
            model.load_state_dict(state_dict)
    if isinstance(model, FSDP):
        model._assert_state(TrainingState.IDLE)
    return loss.detach()