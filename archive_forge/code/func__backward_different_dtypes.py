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
def _backward_different_dtypes(self, t1, t2, sparse):
    local_grads = None
    for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:
        with dist_autograd.context() as context_id:
            loss = self._exec_func(exec_mode, torch.add, t1, t2)
            if sparse:
                loss = torch.sparse.sum(loss)
            else:
                loss = loss.sum()
            local_grads = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2)