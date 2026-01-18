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
def _assert_module_states(model: nn.Module, process_group: dist.ProcessGroup, assert_fn: Callable):
    """
    All-gathers module states across ranks and calls ``assert_fn`` on each pair
    of corresponding states from rank 0 and a nonzero rank. For example, if
    ``assert_fn`` is ``self.assertEqual()``, then this checks that all module
    states are equal across ranks.
    """
    named_module_states = [(param_name, param.detach().cpu()) for param_name, param in model.named_parameters()]
    named_module_states += [(buffer_name, buffer.detach().cpu()) for buffer_name, buffer in model.named_buffers()]
    world_size = dist.get_world_size(process_group)
    olist = [None for _ in range(world_size)]
    dist.all_gather_object(olist, named_module_states, group=process_group)
    rank0_states = olist[0]
    for state in olist[1:]:
        for (_, p1), (_, p2) in zip(rank0_states, state):
            assert_fn(p1, p2)