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
def _test_stream_nested_sync(self, dst):
    x = torch.ones(2, 2).to(0)
    y = torch.ones(2, 2).to(0) * 2
    z = torch.ones(2, 2).to(0) * 3
    nested_dst = worker_name((self.rank + 2) % self.world_size)
    ret = rpc.rpc_sync(dst, TensorPipeAgentCudaRpcTest._nested_slow_add_on_user_stream, args=(nested_dst, x, y, z))
    self.assertEqual(ret, 6 * x)