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
def _backward_no_grad_on_tensor(self, t1, t2, sparse):
    with dist_autograd.context() as context_id:
        loss = rpc.rpc_sync(worker_name(self._next_rank()), torch.add, args=(t1, t2))
        if sparse:
            loss = torch.sparse.sum(loss)
        else:
            loss = loss.sum()
        dist_autograd.backward(context_id, [loss], retain_graph=True)
        self.assertIsNone(t1.grad)
        self.assertIsNone(t2.grad)
        loss_local = torch.add(t1, t2)
        if sparse:
            loss_local = torch.sparse.sum(loss_local)
        else:
            loss_local = loss_local.sum()
        loss_local.backward()
        self.assertIsNotNone(t1.grad)
        self.assertIsNotNone(t2.grad)
        t1_grad_before = t1.grad
        t2_grad_before = t2.grad
        dist_autograd.backward(context_id, [loss])
        self.assertEqual(t1_grad_before, t1.grad)
        self.assertEqual(t2_grad_before, t2.grad)