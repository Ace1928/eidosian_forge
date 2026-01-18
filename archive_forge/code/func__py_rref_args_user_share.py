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
def _py_rref_args_user_share(self, a, b, c, x, y, z, expected):
    n = self.rank + 1
    owner_rank = n % self.world_size
    user_rank = (n + 1) % self.world_size
    rref_a = rpc.remote(worker_name(owner_rank), my_function, args=(a, b, c))
    rref_b = rpc.remote(worker_name(owner_rank), my_function, args=(x, y, z))
    rref_c = rpc.remote(worker_name(user_rank), my_rref_function, args=(rref_a, rref_b))
    self.assertEqual(rref_c.to_here(), expected)