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
def _test_device_maps_return_to_gpu(self, dst):
    options = self.rpc_backend_options
    options.set_device_map(dst, {0: 1})
    options.set_device_map(dst, {1: 2})
    options.set_device_map(dst, {2: 3})
    options.set_device_map(dst, {3: 0})
    rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
    rets = rpc.rpc_sync(dst, TensorPipeAgentCudaRpcTest._gpu_add_return_to_gpu, args=(torch.zeros(2), torch.ones(2)))
    for i in range(len(rets)):
        self.assertEqual(rets[i].device, torch.device((3 + i) % 4))
    self.assertEqual(rets[0], (torch.zeros(2) + torch.ones(2)).to(3))
    self.assertEqual(rets[1], (torch.zeros(2) - torch.ones(2)).to(0))
    self.assertEqual(rets[2], (torch.zeros(2) * torch.ones(2)).to(1))
    self.assertEqual(rets[3], (torch.zeros(2) / torch.ones(2)).to(2))
    rpc.shutdown()