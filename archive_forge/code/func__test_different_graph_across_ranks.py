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
def _test_different_graph_across_ranks(self, find_unused_parameters=False, static_graph=False):

    class ToyModel(nn.Module):

        def __init__(self, rank):
            super().__init__()
            self.lin1 = nn.Linear(10, 10, bias=False)
            self.lin2 = nn.Linear(10, 10, bias=False)
            self.rank = rank

        def forward(self, x):
            if self.rank == 0:
                return self.lin2(F.relu(self.lin1(x)))
            else:
                return F.relu(self.lin1(x))
    torch.manual_seed(31415)
    world_size = dist.get_world_size()
    torch.cuda.set_device(self.rank)
    model = ToyModel(self.rank).cuda(self.rank)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.rank], find_unused_parameters=find_unused_parameters, gradient_as_bucket_view=True, static_graph=static_graph)
    random_input = torch.randn(20, 10, device=self.rank)
    for i in range(10):
        out = ddp_model(random_input)
        loss = out.sum()
        loss.backward()
    return ddp_model