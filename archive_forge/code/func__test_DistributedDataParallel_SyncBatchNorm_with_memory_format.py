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
def _test_DistributedDataParallel_SyncBatchNorm_with_memory_format(self, memory_format):
    group, group_id, rank = self._init_global_test()
    num_processes = dist.get_world_size()
    local_bs = 2
    bs_offset = int(rank * 2)
    global_bs = int(num_processes * 2)
    model = ONLY_SBN_NET
    model_gpu = copy.deepcopy(model).cuda(rank)
    model_DDP = nn.parallel.DistributedDataParallel(model_gpu, device_ids=[rank])
    shapes = [global_bs, 2, 4, 4] + ([] if memory_format is torch.channels_last else [4])
    input_gpu = torch.randn(*shapes, dtype=torch.float).cuda(rank).to(memory_format=memory_format)
    target_gpu = torch.randn(*shapes, dtype=torch.float).cuda(rank).to(memory_format=memory_format)
    loss = nn.MSELoss()
    self._test_DDP_niter(model_gpu, model_DDP, input_gpu, target_gpu, loss, local_bs, rank, global_bs, True, bs_offset, dist.get_world_size(), memory_format=memory_format)
    self._barrier()