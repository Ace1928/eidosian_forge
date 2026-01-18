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
def _run_uneven_workload(self, f, x, num_repeat=30):
    if self.rank == 0:
        self.assertTrue(self.world_size >= 3)
        dst = 'worker1'
        futs = []
        for _ in range(num_repeat):
            fut = rpc.rpc_async(dst, f, args=(x,))
            futs.append(fut)
        for fut in torch.futures.collect_all(futs).wait():
            self.assertEqual(fut.wait(), 0)
        dst = 'worker2'
        futs = []
        for _ in range(num_repeat):
            fut = rpc.rpc_async(dst, f, args=(x,))
            futs.append(fut)
        for val in torch.futures.wait_all(futs):
            self.assertEqual(val, 0)