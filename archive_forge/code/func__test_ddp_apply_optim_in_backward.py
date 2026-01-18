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
def _test_ddp_apply_optim_in_backward(self, optim_cls, optim_kwargs, init_before, gradient_as_bucket_view=True):
    torch.manual_seed(self.rank)
    torch.cuda.manual_seed(self.rank)
    torch.cuda.set_device(self.rank)
    models_to_test = [nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3), nn.Linear(3, 3)).cuda()]
    if HAS_TORCHVISION:
        models_to_test.append(torchvision.models.resnet50().cuda())
    for j, model in enumerate(models_to_test):
        model_optim_in_bwd = copy.deepcopy(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[self.rank], gradient_as_bucket_view=gradient_as_bucket_view)
        optim = optim_cls(model.parameters(), **optim_kwargs)
        if init_before:
            _apply_optimizer_in_backward(optimizer_class=optim_cls, params=model_optim_in_bwd.parameters(), optimizer_kwargs=optim_kwargs)
        model_optim_in_bwd = nn.parallel.DistributedDataParallel(model_optim_in_bwd, device_ids=[self.rank], gradient_as_bucket_view=gradient_as_bucket_view)
        if not init_before:
            _apply_optimizer_in_backward(optimizer_class=optim_cls, params=model_optim_in_bwd.parameters(), optimizer_kwargs=optim_kwargs)
        for p1, p2 in zip(model.parameters(), model_optim_in_bwd.parameters()):
            self.assertEqual(p1, p2, 'Parameters not initially equal!')
        with torch.backends.cudnn.flags(enabled=True, deterministic=True, benchmark=False):
            for i in range(8):
                inp = torch.randn(1, 3, 1000, 1000, device='cuda') if j == 1 else torch.randn(10, 3, device='cuda')
                model(inp).sum().backward()
                optim.step()
                model_optim_in_bwd(inp).sum().backward()
                for p1, p2 in zip(model.parameters(), model_optim_in_bwd.parameters()):
                    self.assertEqual(p1, p2, f'Params not equal at iteration {i}')
                    self.assertTrue(p2.grad is None, f'Optim in backward grad is not None at {i}')
                optim.zero_grad(set_to_none=True)