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
def _test_accumulate_gradients_no_sync(self, num_iters=2, ddp_comm_hook=None, gradient_as_bucket_view=False):
    """
            This is the recommended way to implement accumulate grads.
            If ``ddp_comm_hook`` input was specified, it will also register that hook
            to the ``ddp_model``. The hook fed into this function should not change
            the resulting gradients.
            """
    group, group_id, rank = self._init_global_test()
    world_size = get_world_size()
    if BACKEND == 'mpi' or BACKEND == 'gloo':
        global_batch_size = world_size
        local_batch_size = 1
        model, ddp_model, input, target = self._prepare_cpu_module(group_id, global_batch_size, gradient_as_bucket_view)
    if BACKEND == 'nccl':
        rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
        int_devices = rank_to_GPU[rank][:1]
        devices = [torch.device('cuda:' + str(i)) for i in int_devices]
        global_batch_size = world_size
        local_batch_size = len(devices)
        model, ddp_model, input, target = self._prepare_single_device_module(rank, group_id, devices, devices, global_batch_size, gradient_as_bucket_view)
    if ddp_comm_hook is not None:
        ddp_model.register_comm_hook(group_id, ddp_comm_hook)

    def step_model(model, input, target):
        model.train()
        output = model(input)
        loss = F.mse_loss(output, target.to(output.device))
        loss.backward()
    with torch.no_grad():
        with ddp_model.no_sync():
            ddp_model.train()
            ddp_model(input)
    for iteration in range(num_iters):
        step_model(model, input, target)
        ddp_input = input[rank * local_batch_size:(rank + 1) * local_batch_size]
        ddp_target = target[rank * local_batch_size:(rank + 1) * local_batch_size]
        if iteration % 2 == 0:
            with ddp_model.no_sync():
                step_model(ddp_model, ddp_input, ddp_target)
        else:
            step_model(ddp_model, ddp_input, ddp_target)
        for i, j in zip(model.parameters(), ddp_model.parameters()):
            if not i.requires_grad:
                continue
            if iteration % 2 == 0:
                self.assertNotEqual(i.grad, j.grad)
            else:
                self.assertEqual(i.grad, j.grad)
        torch.manual_seed(1337 + iteration)
        input = input[torch.randperm(global_batch_size)]