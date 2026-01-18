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
def _test_gather_object(self, pg=None):
    gather_objects = COLLECTIVES_OBJECT_TEST_LIST.copy()
    my_rank = dist.get_rank(pg)
    backend = os.environ['BACKEND']
    if backend == 'nccl':
        next_rank = (self.rank + 1) % int(self.world_size)
        torch.cuda.set_device(next_rank)
    if backend == 'nccl':
        gather_objects.append(Foo(torch.randn(3, 3, device=my_rank)))
    output_gathered = [None for _ in range(dist.get_world_size(pg))]
    gather_on_rank = 0
    dist.gather_object(gather_objects[self.rank % len(gather_objects)], object_gather_list=output_gathered if my_rank == gather_on_rank else None, dst=gather_on_rank, group=pg)
    if my_rank != gather_on_rank:
        self.assertEqual(output_gathered, [None for _ in range(dist.get_world_size())])
    else:
        for i, val in enumerate(output_gathered):
            expected = gather_objects[i % len(gather_objects)]
            self.assertEqual(val, expected)

    class Bar:
        pass
    b = Bar()
    gather_objects = [b for _ in range(dist.get_world_size())]
    with self.assertRaisesRegex(AttributeError, "Can't pickle local object"):
        dist.all_gather_object([None for _ in range(dist.get_world_size())], gather_objects[self.rank], group=pg)