import concurrent.futures
import contextlib
import json
import os
import sys
import threading
import time
from collections import namedtuple
from functools import partial
from threading import Event
from threading import Lock
from unittest import mock
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.rpc import RRef, _get_debug_info, _rref_context_get_debug_info, WorkerInfo
from torch.distributed.rpc.api import _use_rpc_pickler, _thread_local_var, _wait_all
from torch.distributed.rpc.internal import (
from torch.futures import Future
from torch.testing._internal.common_distributed import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_utils import TemporaryFileName
from torch.autograd.profiler_legacy import profile as _profile
def _test_test_async_class_rref_proxy(self, mode=RPCExecMode.SYNC):
    dst1 = worker_name((self.rank + 1) % self.world_size)
    dst2 = worker_name((self.rank + 2) % self.world_size)
    rref = rpc.remote(dst1, AsyncExecutionClass)
    x = torch.ones(2, 2)
    y = torch.ones(2, 2) + 1
    if mode == RPCExecMode.SYNC:
        ret = rref.rpc_sync().static_async_add(dst2, x, x, y)
        ret += rref.rpc_sync().class_async_add(dst2, x, x, y)
        ret += rref.rpc_sync().bound_async_add(dst2, x, x, y)
    elif mode == RPCExecMode.ASYNC:
        ret = rref.rpc_async().static_async_add(dst2, x, x, y).wait()
        ret += rref.rpc_async().class_async_add(dst2, x, x, y).wait()
        ret += rref.rpc_async().bound_async_add(dst2, x, x, y).wait()
    elif mode == RPCExecMode.REMOTE:
        ret = rref.remote().static_async_add(dst2, x, x, y).to_here()
        ret += rref.remote().class_async_add(dst2, x, x, y).to_here()
        ret += rref.remote().bound_async_add(dst2, x, x, y).to_here()
    self.assertEqual(ret, 3 * 4 * x)