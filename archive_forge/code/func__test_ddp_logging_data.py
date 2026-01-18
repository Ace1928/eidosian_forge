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
def _test_ddp_logging_data(self, is_gpu):
    rank = dist.get_rank()
    model_DDP = copy.deepcopy(DDP_NET)
    if is_gpu:
        model_DDP = nn.parallel.DistributedDataParallel(model_DDP.cuda(rank), device_ids=[rank])
    else:
        model_DDP = nn.parallel.DistributedDataParallel(model_DDP)
    local_bs = 2
    batch_size, input, target, loss = self._prepare_dummy_data(local_bs)
    if is_gpu:
        input = input.cuda(rank)
        target = target.cuda(rank)
    model_DDP._set_ddp_runtime_logging_sample_rate(2)
    for idx in range(20):
        offset = rank * local_bs
        self._test_DDP_helper(model_DDP, input[offset:offset + local_bs], target[offset:offset + local_bs], loss, 1)
        self._model_step_with_zero_grad(model_DDP)
        ddp_logging_data = model_DDP._get_ddp_logging_data()
        if idx > 0 and (idx < 10 or idx % 2 == 0):
            self.assertGreaterEqual(ddp_logging_data.get('forward_compute_time'), 1)
            self.assertGreaterEqual(ddp_logging_data.get('backward_compute_time'), 1)
            self.assertGreaterEqual(ddp_logging_data.get('backward_comm_time'), 1)
            self.assertGreaterEqual(ddp_logging_data.get('backward_compute_time'), ddp_logging_data.get('backward_compute_comm_overlap_time'))
            self.assertGreaterEqual(ddp_logging_data.get('backward_comm_time'), ddp_logging_data.get('backward_compute_comm_overlap_time'))
            self.assertEqual(ddp_logging_data.get('iteration'), idx)
        elif idx > 0:
            self.assertNotEqual(ddp_logging_data.get('iteration'), idx)
        input = input[torch.randperm(batch_size)]
    return model_DDP