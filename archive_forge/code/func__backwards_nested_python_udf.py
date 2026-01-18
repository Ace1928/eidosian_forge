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
def _backwards_nested_python_udf(self, t1, t2, sparse):
    t3 = t1 * t2
    t4 = t1 + t2
    res = t3 + t4
    loss = t1 * t2 * t3 * t4 * res
    if sparse:
        loss = torch.sparse.sum(loss)
    else:
        loss = loss.sum()
    torch.autograd.backward([loss])
    with dist_autograd.context() as context_id:
        loss = rpc.rpc_sync(worker_name(self._next_rank()), DistAutogradTest._nested_python_udf, args=(t1, t2, self._next_rank()))
        if sparse:
            loss = torch.sparse.sum(loss)
        else:
            loss = loss.sum()
        dist_autograd.backward(context_id, [loss])
        grads = dist_autograd.get_gradients(context_id)
        self.assertEqual(t1.grad, grads[t1])
        self.assertEqual(t2.grad, grads[t2])