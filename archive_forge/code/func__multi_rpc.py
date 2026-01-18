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
def _multi_rpc(self, sparse):
    dst_rank = (self.rank + 1) % self.world_size
    for i in range(20):
        n = i + self.rank + 1
        if sparse:
            x = build_sparse_tensor() * n
            y = build_sparse_tensor() * n
        else:
            x = torch.ones(2, 2)
            y = torch.ones(2, 2)
        ret = rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(x, y))
        self.assertEqual(ret, x * 2)