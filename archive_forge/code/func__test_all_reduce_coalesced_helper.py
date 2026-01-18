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
def _test_all_reduce_coalesced_helper(self, group, group_id, rank, op, cuda=False, rank_to_GPU=None):
    test_case_func = {dist.ReduceOp.SUM: self._all_reduce_coalesced_sum_test_cases, dist.ReduceOp.PRODUCT: self._all_reduce_coalesced_product_test_cases, dist.ReduceOp.MIN: self._all_reduce_coalesced_min_test_cases, dist.ReduceOp.MAX: self._all_reduce_coalesced_max_test_cases}[op]
    master_values, worker_values, expected_values, dtypes = test_case_func(len(group))
    for src in group:
        curr_values = master_values if rank == src else worker_values
        tensors = [_build_tensor(src + 1, val, dtype=dtype) for dtype, val in zip(dtypes, curr_values)]
        if cuda:
            tensors = [t.cuda(rank_to_GPU[rank][0]) for t in tensors]
        tensor_shapes = []
        for tensor in tensors:
            if tensor.dtype == torch.complex64:
                tensor_shapes.append(torch.view_as_real(tensor).shape)
            else:
                tensor_shapes.append(tensor.shape)
        self.call_dist_op(':all_reduce', False, dist.all_reduce_coalesced, tensors, op, group_id, tensor_shapes=tensor_shapes)
        expected_tensors = [_build_tensor(src + 1, expected_value, dtype=dtype) for dtype, expected_value in zip(dtypes, expected_values)]
        self.assertEqual(tensors, expected_tensors)
    self._barrier()