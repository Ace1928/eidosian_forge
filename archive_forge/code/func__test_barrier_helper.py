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
def _test_barrier_helper(self, info, names, multi_threaded=False):
    names = sorted(names)
    leader = names[0]
    rpc.rpc_sync(leader, _reset_count)
    if not multi_threaded and info.name == leader:
        self.assertEqual(_rpc_barrier_count, 0)
    rpc.api._barrier(names)
    rpc.rpc_sync(leader, _increment_count)
    rpc.api._barrier(names)
    if not multi_threaded and info.name == leader:
        self.assertEqual(_rpc_barrier_count, len(names))