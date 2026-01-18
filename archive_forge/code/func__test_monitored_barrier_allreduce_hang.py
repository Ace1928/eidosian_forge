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
def _test_monitored_barrier_allreduce_hang(self, wait_all_ranks):
    nccl_pg = dist.new_group(ranks=list(range(int(self.world_size))), timeout=timedelta(seconds=15), backend=dist.Backend.NCCL)
    gloo_pg = dist.new_group(ranks=list(range(int(self.world_size))), backend=dist.Backend.GLOO)
    tensors = [torch.ones(10, device=self.rank) * self.rank]
    nccl_pg.allreduce(tensors).wait(timedelta(seconds=5))
    if self.rank != 0:
        if dist.get_debug_level() == dist.DebugLevel.DETAIL:
            err_regex = 'Timed out waiting'
        else:
            err_regex = 'caught collective operation timeout'
        with self.assertRaisesRegex(RuntimeError, err_regex):
            nccl_pg.allreduce(tensors).wait(timedelta(seconds=0.1))
    else:
        if wait_all_ranks:
            rank_str = ', '.join([str(i) for i in range(1, int(self.world_size))])
            err_regex = f'Ranks {rank_str} failed to pass monitoredBarrier'
        else:
            expected_first_fail_rank = 1
            err_regex = f'Rank {expected_first_fail_rank} failed to pass monitoredBarrier'
        monitored_barrier_timeout_seconds = timedelta(seconds=0.1)
        with self.assertRaisesRegex(RuntimeError, err_regex):
            gloo_pg.monitored_barrier(monitored_barrier_timeout_seconds, wait_all_ranks=wait_all_ranks)
    self._barrier(timeout=30)