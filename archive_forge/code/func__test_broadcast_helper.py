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
def _test_broadcast_helper(self, group, group_id, rank, cuda=False, rank_to_GPU=None, with_options=False):
    for dtype, value, requires_cuda in [(torch.float, -1e-10, False), (torch.double, -1e-100, False), (torch.half, -0.1, True), (torch.int8, -2, False), (torch.uint8, 129, False), (torch.int, -100000.0, False), (torch.long, -1000000000000000.0, False)]:
        if requires_cuda and (not cuda):
            continue
        for src in group:
            expected_tensor = _build_tensor(src + 1, value, dtype)
            if cuda:
                expected_tensor = expected_tensor.cuda(rank_to_GPU[rank][0])
            if rank == src:
                if with_options:
                    opts = dist.BroadcastOptions()
                    opts.rootTensor = 0
                    opts.rootRank = src
                    self.call_dist_op(':broadcast', True, group_id.broadcast, [expected_tensor], opts)
                else:
                    self.call_dist_op(':broadcast', False, dist.broadcast, expected_tensor, src, group_id)
            else:
                tensor = _build_tensor(src + 1, -1, dtype)
                if cuda:
                    tensor = tensor.cuda(rank_to_GPU[rank][0])
                if with_options:
                    opts = dist.BroadcastOptions()
                    opts.rootTensor = 0
                    opts.rootRank = src
                    self.call_dist_op(':broadcast', True, group_id.broadcast, [tensor], opts)
                else:
                    self.call_dist_op(':broadcast', False, dist.broadcast, tensor, src, group_id)
                self.assertEqual(tensor.size(), expected_tensor.size())
                self.assertEqual(tensor.ne(expected_tensor).max(), torch.tensor(False))
    self._barrier()