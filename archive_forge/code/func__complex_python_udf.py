import sys
import threading
import time
from enum import Enum
import random
import torch
import torch.nn as nn
from datetime import timedelta
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributed.rpc import RRef
from torch.testing._internal.common_utils import IS_MACOS, skip_but_pass_in_sandcastle_if
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
@staticmethod
def _complex_python_udf(t1, t2):
    t3 = torch.nn.functional.linear(t1, t2)
    t4 = torch.nn.functional.linear(t2, t3)
    t5 = torch.nn.functional.linear(t3, t4)
    return torch.linalg.multi_dot([t1, t2, t3, t4, t5])