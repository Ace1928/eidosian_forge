import copy
import itertools
import math
import os
import random
import sys
import tempfile
import time
from collections import namedtuple, OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from functools import reduce
from typing import Union, NamedTuple, Callable, Any
import unittest
import numpy as np
import torch
import torch.cuda
import torch.distributed as dist
import torch.distributed.algorithms.model_averaging.averagers as averagers
import torch.distributed.algorithms.model_averaging.hierarchical_model_averager as hierarchicalSGD
import torch.distributed.algorithms.model_averaging.utils as model_averaging_utils
import torch.nn as nn
import torch.nn.functional as F
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT
from torch.cuda.amp import GradScaler, autocast
from torch.distributed.algorithms.ddp_comm_hooks import (
from torch.distributed.optim import _apply_optimizer_in_backward
from torch.distributed.distributed_c10d import (
from torch.distributed.utils import (
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.distributed import _dump_DDP_relevant_env_vars, _MixedPrecision
from torch.testing._internal.common_distributed import (
from torch.testing._internal.common_utils import (
import torch.distributed.optim.post_localSGD_optimizer as post_localSGD_optimizer
from torch.utils.data.distributed import DistributedSampler
def _test_ddp_ignore_params_arg(self, static_graph=False):

    class TestModel(nn.Module):

        def __init__(self, rank):
            self.rank = rank
            super().__init__()
            self.fc1 = nn.Linear(1, 1, bias=False)
            if self.rank == 0:
                self.fc2 = nn.Linear(1, 10, bias=False)
            else:
                self.fc2 = nn.Linear(10, 10, bias=False)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x
    device_id = self.rank
    for find_unused, broadcast_buffers in itertools.product([False, True], [False, True]):
        model = TestModel(self.rank).float().to(device_id)
        model.fc2.register_buffer('ignore_buffer', torch.zeros(5 + self.rank, device=self.rank))
        proxy_params = list(model.fc2.parameters())
        proxy_buffers = list(model.fc2.buffers())
        model_fc2_name = next((module_name for module_name, module in model.named_modules() if module is model.fc2))
        proxy_param_names = [f'{model_fc2_name}.{param_name}' for param_name, _ in model.fc2.named_parameters()]
        proxy_buffer_names = [f'{model_fc2_name}.{buf_name}' for buf_name, _ in model.fc2.named_buffers()]
        torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, proxy_param_names + proxy_buffer_names)
        ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], find_unused_parameters=find_unused, broadcast_buffers=broadcast_buffers, static_graph=static_graph)
        ddp.module.fc2 = nn.Linear(1, 1, bias=False).to(device_id)
        local_model = copy.deepcopy(ddp.module).cuda(self.rank)
        inp = torch.ones(1, dtype=torch.float).to(device_id) * (self.rank + 1)
        for i in range(6):
            ddp(inp).sum().backward()
            local_model(inp).sum().backward()
            for materialized_param, local_param in zip(ddp.module.fc2.parameters(), local_model.fc2.parameters()):
                self.assertEqual(materialized_param.grad, local_param.grad)
            for synced_param, local_param in zip(ddp.module.fc1.parameters(), local_model.fc1.parameters()):
                self.assertFalse(synced_param.grad == local_param.grad)
            for proxy_param in proxy_params:
                self.assertTrue(proxy_param.grad is None)
        torch.cuda.synchronize(device=self.rank)