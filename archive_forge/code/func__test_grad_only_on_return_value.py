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
def _test_grad_only_on_return_value(self, exec_mode):
    initialize_pg(self.file_init_method, self.rank, self.world_size)
    dst_rank = (self.rank + 1) % self.world_size
    with dist_autograd.context() as context_id:
        if ExecMode.RPC_SYNC == exec_mode:
            ret = rpc.rpc_sync(worker_name(dst_rank), ret_requires_grad)
        elif ExecMode.REMOTE == exec_mode:
            ret = rpc.remote(worker_name(dst_rank), ret_requires_grad).to_here()
        else:
            raise ValueError(f'Unrecognized ExecMode {exec_mode}')
        dist_autograd.backward(context_id, [ret.sum()])
        rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))
        self._check_rpc_done(1)
        grads = dist_autograd.get_gradients(ctx_ids[1])
        self.assertEqual(1, len(grads))
        self.assertIn(requires_grad_tensor, grads)
        self.assertEqual(torch.ones_like(ret), grads[requires_grad_tensor])
        dist.barrier()