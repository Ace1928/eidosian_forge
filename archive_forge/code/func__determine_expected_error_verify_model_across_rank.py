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
def _determine_expected_error_verify_model_across_rank(self, group_to_use, diff_num_params=False):
    if diff_num_params:
        expected_err = 'DDP expects same model across all ranks'
        ctx = self.assertRaisesRegex(RuntimeError, expected_err)
        return (ctx, expected_err)
    is_detail_dbg_mode = dist.get_debug_level() == dist.DebugLevel.DETAIL
    if self.rank == 0:
        if dist.get_backend(group_to_use) == dist.Backend.NCCL and (not is_detail_dbg_mode):
            expected_err = 'caught collective operation timeout'
            ctx = self.assertRaisesRegex(RuntimeError, expected_err)
        else:
            expected_err = None
            ctx = self.assertRaises(RuntimeError)
    else:
        expected_err = 'appears not to match'
        ctx = self.assertRaisesRegex(RuntimeError, expected_err)
    return (ctx, expected_err)