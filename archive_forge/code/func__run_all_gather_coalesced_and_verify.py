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
def _run_all_gather_coalesced_and_verify(self, output_tensor_lists, input_tensors, expected_tensors, group_id):
    """
            Helper that runs all_gather_coalesced and returns true if output
            matches expectations.
            """
    tensor_shapes = []
    for input_tensor in input_tensors:
        if input_tensor.dtype == torch.complex64:
            tensor_shapes.append(torch.view_as_real(input_tensor).shape)
        else:
            tensor_shapes.append(input_tensor.shape)
    self.call_dist_op(':all_gather', False, dist.all_gather_coalesced, output_tensor_lists, input_tensors, group_id, tensor_shapes=tensor_shapes)
    for l1, l2 in zip(output_tensor_lists, expected_tensors):
        for t1, t2 in zip(l1, l2):
            if not torch.equal(t1, t2):
                return False
    return True