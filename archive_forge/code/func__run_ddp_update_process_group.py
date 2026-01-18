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
def _run_ddp_update_process_group(self, new_pg):

    def get_num_torch_recompiles():
        guard_failures = torch._dynamo.utils.guard_failures
        num_recompiles = [len(guard_failures[code]) for code in guard_failures]
        return 0 if len(num_recompiles) == 0 else max(num_recompiles)

    class SimulateError(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            return input

        @staticmethod
        def backward(ctx, grad_output):
            raise RuntimeError()

    class MyModel(torch.nn.Module):

        def __init__(self, device):
            super().__init__()
            self.fc1 = torch.nn.Linear(1024, 1024).cuda(device)
            self.fc2 = torch.nn.Linear(1024, 1024).cuda(device)
            self.fc3 = torch.nn.Linear(1024, 1024).cuda(device)

        def forward(self, inp, error):
            if error:
                return self.fc3(self.fc2(self.fc1(SimulateError.apply(inp))))
            else:
                return self.fc3(self.fc2(self.fc1(inp)))
    input = torch.rand(10, 1024, requires_grad=True).cuda(self.rank)
    ddp = torch.nn.parallel.DistributedDataParallel(MyModel(self.rank), device_ids=[self.rank], find_unused_parameters=True, bucket_cap_mb=1)
    model = torch.compile(ddp)

    def run_iteration():
        out = model(input, error=False)
        out.sum().backward()
        torch.cuda.synchronize()
        with self.assertRaises(RuntimeError):
            out = model(input, error=True)
            out.sum().backward()
        torch.cuda.synchronize()
    run_iteration()
    assert 0 == get_num_torch_recompiles()
    if new_pg:
        group_size_2 = dist.new_group(ranks=[0, 1])
        ddp._update_process_group(group_size_2)
        if self.rank in [0, 1]:
            run_iteration()
        group_size_3 = dist.new_group(ranks=[1, 2, 3])
        ddp._update_process_group(group_size_3)
        if self.rank in [1, 2, 3]:
            run_iteration()
        ddp._update_process_group(_get_default_group())
        run_iteration()
    else:
        dist.destroy_process_group()
        if self.rank in [1, 2, 3]:
            dist.init_process_group(init_method=self.init_method, backend=BACKEND, world_size=3, rank=self.rank - 1, timeout=timedelta(seconds=default_pg_timeout))
            ddp._update_process_group(_get_default_group())
            run_iteration()
            dist.destroy_process_group()
        self._barrier(wait_for=4)
        dist.init_process_group(init_method=self.init_method, backend=BACKEND, world_size=4, rank=self.rank, timeout=timedelta(seconds=default_pg_timeout))
    assert 0 == get_num_torch_recompiles()