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
@staticmethod
def _slow_add_on_user_stream(x, y):
    s0 = torch.cuda.current_stream(x.device)
    s1 = torch.cuda.Stream(device=x.device)
    s1.wait_stream(s0)
    x.record_stream(s1)
    y.record_stream(s1)
    with torch.cuda.stream(s1):
        torch.cuda._sleep(10 * FIFTY_MIL_CYCLES)
        z = x + y
    s0.wait_stream(s1)
    z.record_stream(s0)
    return z