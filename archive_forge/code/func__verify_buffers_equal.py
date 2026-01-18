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
def _verify_buffers_equal(self, m1, m2):
    m1_buf_dict = dict(m1.module.named_buffers())
    for name, buf in m2.module.named_buffers():
        self.assertEqual(buf, m1_buf_dict[name])
    m1_buffers = list(m1.buffers())
    m2_buffers = list(m2.buffers())
    for buf1, buf2 in zip(m1_buffers, m2_buffers):
        gathered_bufs = [torch.empty_like(buf1) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_bufs, buf1)
        gathered_bufs_m2 = [torch.empty_like(buf2) for _ in range(dist.get_world_size())]
        for b in gathered_bufs:
            self.assertEqual(b, buf1)
        dist.all_gather(gathered_bufs_m2, buf2)
        for b in gathered_bufs_m2:
            self.assertEqual(b, buf2)