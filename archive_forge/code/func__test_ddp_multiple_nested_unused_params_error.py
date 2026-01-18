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
def _test_ddp_multiple_nested_unused_params_error(self, ignore_sparse):
    debug_mode_off = dist.get_debug_level() == dist.DebugLevel.OFF

    class SubModule(nn.Module):

        def __init__(self):
            super().__init__()
            self.embedding_net = EmbeddingNetDifferentParams(0)
            self.lin = TwoLinLayerNet()
            self.bn = BatchNormNet()
            self.lin_layer = nn.Linear(4, 10, bias=False)

        def forward(self, x):
            x = self.bn(x)
            x = self.lin_layer(x)
            x = self.lin.a(x)
            return x

    class MyModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.sub_module = SubModule()

        def forward(self, x):
            return self.sub_module(x)
    model = MyModel()
    sparse_embedding_fqns = []
    if ignore_sparse:
        for module_name, module in model.named_modules():
            if module == model.sub_module.embedding_net.embedding:
                for parameter_name, param in module.named_parameters(recurse=False):
                    fqn = f'{module_name}.{parameter_name}'
                    sparse_embedding_fqns.append(fqn)
        torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, sparse_embedding_fqns)
        unused_modules = [model.sub_module.embedding_net.lin, model.sub_module.lin.b]
    else:
        unused_modules = list(model.sub_module.embedding_net.modules()) + [model.sub_module.lin.b]
    expected_unused_param_fqns = []
    used_param_fqns = []
    fqn_to_param_index = {}
    index = 0
    for module_name, module in model.named_modules():
        for parameter_name, param in module.named_parameters(recurse=False):
            fqn = f'{module_name}.{parameter_name}'
            fqn_to_param_index[fqn] = index
            if fqn not in sparse_embedding_fqns:
                index += 1
            if module in unused_modules:
                expected_unused_param_fqns.append(fqn)
            elif not ignore_sparse or module != model.sub_module.embedding_net.embedding:
                used_param_fqns.append(fqn)
    net = torch.nn.parallel.DistributedDataParallel(model.cuda(self.rank), device_ids=[self.rank])
    batch, dim = (10, 2)
    inp = torch.ones(batch, dim)
    for i in range(2):
        if i == 0:
            out = net(inp)
            loss = out.sum()
            loss.backward()
        else:
            try:
                out = net(inp)
                loss = out.sum()
                loss.backward()
            except RuntimeError as e:
                e = str(e)
                unused_param_substr = e[e.find('did not receive grad'):]
                for unused_param_fqn in expected_unused_param_fqns:
                    self.assertTrue(unused_param_fqn in unused_param_substr or debug_mode_off)
                    self.assertTrue(str(fqn_to_param_index[unused_param_fqn]) in unused_param_substr, f'Did not find index {fqn_to_param_index[unused_param_fqn]} for {unused_param_fqn}')
                for used_param_fqn in used_param_fqns:
                    self.assertFalse(used_param_fqn in unused_param_substr)
                for sparse_param_fqn in sparse_embedding_fqns:
                    self.assertFalse(sparse_param_fqn in unused_param_substr)
            else:
                self.assertTrue(False, 'Expected error was not raised!')