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
def _test_ddp_profiling(self, profiler_ctx):
    batch = 3
    dim = 10
    num_iters = 6
    torch.cuda.set_device(self.rank)
    model = nn.Linear(dim, dim, bias=False)
    inp = torch.rand(batch, dim, device=self.rank)
    net = torch.nn.parallel.DistributedDataParallel(model.cuda(self.rank), device_ids=[self.rank])
    profiler_ctx_copy = copy.deepcopy(profiler_ctx)
    with profiler_ctx as prof:
        for i in range(num_iters):
            loss = net(inp).sum()
            loss.backward()
    all_reduce_event_name = f'{dist.get_backend()}:all_reduce'
    events = get_profiling_event(all_reduce_event_name, prof)
    event_count = sum((e.count for e in events))
    self.assertEqual(event_count, num_iters)
    for event in events:
        self.assertTrue(event.is_async)
        self.assertEqual(event.name, all_reduce_event_name)
    broadcast_event_name = f'{dist.get_backend()}:broadcast'
    broadcast_events = get_profiling_event(broadcast_event_name, prof)
    event_count = sum((e.count for e in broadcast_events))
    self.assertGreaterEqual(event_count, 1)
    for event in broadcast_events:
        self.assertEqual(event.name, broadcast_event_name)
    net = torch.nn.parallel.DistributedDataParallel(model.cuda(self.rank), device_ids=[self.rank], find_unused_parameters=True)
    for i in range(3):
        loss = net(inp).sum()
        loss.backward()
    with profiler_ctx_copy as prof:
        loss = net(inp).sum()
        loss.backward()
    events = get_profiling_event(all_reduce_event_name, prof)
    self.assertGreaterEqual(len(events), 1)
    self.assertGreaterEqual(events[0].count, 1)
    self.assertEqual(events[0].name, all_reduce_event_name)
    for event in events:
        self.assertTrue(event.is_async)
    events = get_profiling_event('search_unused_parameters', prof)
    self.assertEqual(len(events), 1)