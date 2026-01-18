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
def _run_rpc_profiling_async_function(self, device='cpu'):
    if self.rank != 1:
        return
    dst1 = worker_name((self.rank + 1) % self.world_size)
    dst2 = worker_name((self.rank + 2) % self.world_size)
    x = torch.ones(2)
    y = torch.ones(2)
    with _profile() as prof:
        ret = rpc.rpc_async(dst1, slow_async_add, args=(dst2, x, y, device), timeout=20)
        out = ret.wait()
    function_events = prof.function_events
    key_prefix = _build_rpc_profiling_key(RPCExecMode.ASYNC, slow_async_add.__qualname__, worker_name(self.rank), dst1)
    nested_rpc_key_prefix = _build_rpc_profiling_key(RPCExecMode.ASYNC, slow_add.__qualname__, dst1, dst2)
    expected_key = key_prefix + REMOTE_OP_STR + nested_rpc_key_prefix
    remote_events = [event for event in function_events if event.is_remote]
    rpc_remote_event = [event for event in remote_events if event.name == expected_key]
    self.assertEqual(1, len(rpc_remote_event))
    rpc_remote_event = rpc_remote_event[0]
    self.assertEqual(rpc_remote_event.node_id, (self.rank + 1) % self.world_size)
    remote_add_key = expected_key + REMOTE_OP_STR + torch.jit._builtins._find_builtin(torch.add)
    remote_add_event = [event for event in remote_events if event.name == remote_add_key]
    self.assertEqual(1, len(remote_add_event))
    remote_add_event = remote_add_event[0]
    self.assertEqual(remote_add_event.node_id, (self.rank + 2) % self.world_size)