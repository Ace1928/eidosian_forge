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
def _test_send_recv_with_tag(self, profiler_ctx):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    send_recv_size = 10
    tensor = _build_tensor(send_recv_size, value=rank)
    ctx = profiler_ctx if profiler_ctx is not None else nullcontext()
    with ctx as prof:
        for dst in range(0, world_size):
            if dst == rank:
                for src in range(0, world_size):
                    if src == rank:
                        continue
                    output_tensor = _build_tensor(send_recv_size, value=-1)
                    dist.recv(output_tensor, src, tag=src)
                    self.assertTrue(output_tensor.eq(src).all())
            else:
                dist.send(tensor, dst, tag=rank)
    if profiler_ctx is not None:
        backend = dist.get_backend()
        if backend in SEND_RECV_PROFILING_SUPPORTED_BACKENDS:
            for event_name in [f'{backend}:send', f'{backend}:recv']:
                events = get_profiling_event(event_name, prof)
                event_count = sum((e.count for e in events))
                expected_event_count = dist.get_world_size() - 1
                self.assertEqual(event_count, expected_event_count)
                for event in events:
                    self.assertTrue(event.is_async)
                    self.assertEqual(event.name, event_name)
                    self.assertEqual(event.input_shapes, [[send_recv_size] * 3])