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
def _exec_func_with_dst(self, dst, exec_mode, method, *args):
    if ExecMode.LOCAL == exec_mode:
        if len(args) == 1 and isinstance(args[0], list):
            return method(*args[0])
        return method(*args)
    elif ExecMode.RPC_SYNC == exec_mode:
        return rpc.rpc_sync(worker_name(dst), method, args=args)
    elif ExecMode.REMOTE == exec_mode:
        return rpc.remote(worker_name(dst), method, args=args).to_here()
    elif ExecMode.RPC_ASYNC == exec_mode:
        fut = rpc.rpc_async(worker_name(dst), method, args=args)
        return fut.wait()
    else:
        raise ValueError(f'Unrecognized ExecMode {exec_mode}')