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
def _test_broadcast_object_list(self, group=None):
    gather_objects = COLLECTIVES_OBJECT_TEST_LIST.copy()
    next_rank = (self.rank + 1) % int(self.world_size)
    backend = os.environ['BACKEND']
    if backend == 'nccl':
        torch.cuda.set_device(next_rank)
    src_rank = 0
    if backend == 'nccl':
        gather_objects.append(Foo(torch.randn(3, 3, device=0)))
    if IS_FBCODE:
        gather_objects.append(Foo(torch.randn(3, 178956971)))
    objects = gather_objects if self.rank == src_rank else [None for _ in gather_objects]
    if backend != 'nccl':
        single_obj_list = [objects[0]]
        if self.rank != src_rank:
            self.assertNotEqual(single_obj_list[0], gather_objects[0])
        dist.broadcast_object_list(single_obj_list, src=0, group=group, device=torch.device('cpu'))
        self.assertEqual(single_obj_list[0], gather_objects[0])
    if backend != 'nccl' and torch.cuda.device_count() == int(self.world_size):
        single_obj_list = [objects[0]]
        if self.rank != src_rank:
            self.assertNotEqual(single_obj_list[0], gather_objects[0])
        dist.broadcast_object_list(single_obj_list, src=0, group=group, device=torch.device(next_rank))
        self.assertEqual(single_obj_list[0], gather_objects[0])
    if backend == 'nccl' and torch.cuda.device_count() == int(self.world_size):
        single_obj_list = [objects[0]]
        if self.rank != src_rank:
            self.assertNotEqual(single_obj_list[0], gather_objects[0])
        dist.broadcast_object_list(single_obj_list, src=0, group=group, device=torch.device(next_rank))
        self.assertEqual(single_obj_list[0], gather_objects[0])
    single_obj_list = [objects[0]]
    if self.rank != src_rank:
        self.assertNotEqual(single_obj_list[0], gather_objects[0])
    dist.broadcast_object_list(single_obj_list, src=0, group=group)
    self.assertEqual(single_obj_list[0], gather_objects[0])
    if self.rank != src_rank:
        self.assertNotEqual(objects, gather_objects)
    dist.broadcast_object_list(objects, src=0, group=group)
    self.assertEqual(objects, gather_objects)