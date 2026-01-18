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
def _test_graph(self, fn, exec_mode, sparse):
    dst_rank = (self.rank + 1) % self.world_size
    initialize_pg(self.file_init_method, self.rank, self.world_size)
    with dist_autograd.context() as context_id:
        if sparse:
            t1 = build_sparse_tensor()
            t2 = build_sparse_tensor()
        else:
            t1 = torch.ones(3, 3, requires_grad=True)
            t2 = torch.zeros(3, 3, requires_grad=True)
        if ExecMode.RPC_SYNC == exec_mode:
            ret = rpc.rpc_sync(worker_name(dst_rank), fn, args=(t1, t2))
        elif ExecMode.REMOTE == exec_mode:
            ret = rpc.remote(worker_name(dst_rank), fn, args=(t1, t2)).to_here()
        else:
            raise ValueError(f'Unrecognized ExecMode {exec_mode}')
        rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))
        ctx = dist_autograd._current_context()
        self.assertEqual(context_id, ctx._context_id())
        send_functions = ctx._send_functions()
        self.assertEqual(1, len(send_functions))
        recv_functions = ctx._recv_functions()
        self.assertEqual(1, len(recv_functions))
        self._verify_graph_for_first_rpc_call(next(iter(send_functions.values())), next(iter(recv_functions.values())), t1, t2, ret)
        self._check_rpc_done(1)
        ctx = dist_autograd._retrieve_context(ctx_ids[1])
        send_functions = ctx._send_functions()
        self.assertEqual(1, len(send_functions))
        self._verify_graph_for_rpc_call_exec(next(iter(send_functions.values())))
        dist.barrier()
    with self.assertRaises(RuntimeError):
        ctx = dist_autograd._retrieve_context(context_id)
    with self.assertRaises(RuntimeError):
        ctx = dist_autograd._current_context()