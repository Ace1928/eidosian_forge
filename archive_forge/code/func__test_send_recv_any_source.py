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
def _test_send_recv_any_source(self, profiler_ctx):
    rank = dist.get_rank()
    send_recv_size = 10
    tensor = _build_tensor(send_recv_size, value=rank)
    recv_ranks = list()
    irecv_ranks = list()
    ctx = profiler_ctx if profiler_ctx is not None else nullcontext()
    with ctx as prof:
        for dst in range(0, dist.get_world_size()):
            if dst == rank:
                for dst in range(0, dist.get_world_size()):
                    if dst == rank:
                        continue
                    for recv in ['recv', 'irecv']:
                        output_tensor = _build_tensor(send_recv_size, value=-1)
                        if recv == 'recv':
                            sender = dist.recv(output_tensor)
                            recv_ranks.append(sender)
                        elif recv == 'irecv':
                            work = dist.irecv(output_tensor)
                            work.wait()
                            sender = work._source_rank()
                            irecv_ranks.append(sender)
                        self.assertTrue(output_tensor.eq(sender).all())
            else:
                dist.send(tensor, dst)
                dist.send(tensor, dst)
    if profiler_ctx is not None:
        backend = dist.get_backend()
        if backend in SEND_RECV_PROFILING_SUPPORTED_BACKENDS:
            for event_name in [f'{backend}:send', f'{backend}:recvAnySource']:
                events = get_profiling_event(event_name, prof)
                self.assertEqual(sum((event.count for event in events)), 2 * (dist.get_world_size() - 1))
                for event in events:
                    self.assertTrue(event.is_async)
                    self.assertEqual(event.input_shapes, [[send_recv_size] * 3])
        recv_ranks_tensor = torch.cat((torch.tensor(recv_ranks), torch.tensor(irecv_ranks)), 0)
        global_recv_ranks = [torch.empty_like(recv_ranks_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(global_recv_ranks, recv_ranks_tensor)
        global_recv_ranks_list = []
        for tensor in global_recv_ranks:
            global_recv_ranks_list += tensor.tolist()
        from itertools import groupby
        global_recv_ranks_list.sort()
        frequency = [len(list(group)) for key, group in groupby(global_recv_ranks_list)]
        self.assertEqual(dist.get_world_size(), len(frequency))
        self.assertEqual([2 * (dist.get_world_size() - 1)] * dist.get_world_size(), frequency)
        self._barrier()