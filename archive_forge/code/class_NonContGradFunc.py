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
class NonContGradFunc(Function):
    static_grad_ptr = None

    @staticmethod
    def forward(ctx, inp1, inp2):
        return inp1 + inp2

    @staticmethod
    def backward(ctx, grad):
        v = torch.rand(1, 3)
        i = torch.ones(1, 1, dtype=torch.long)
        nv = v.expand(8, 3)
        ni = i.expand(1, 8)
        ngrad = torch.sparse_coo_tensor(ni, nv, (10, 3), dtype=torch.float32)
        NonContGradFunc.static_grad_ptr = ngrad._values().data_ptr()
        return (ngrad, ngrad)