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
class MyConvNetForMNIST(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 16, 3, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(1), nn.Linear(4608, 128), nn.ReLU(), nn.Linear(128, 10)).to(device)
        self.device = device

    def forward(self, x, is_rref=False):
        x = x.to_here() if is_rref else x
        with torch.cuda.stream(torch.cuda.current_stream(self.device)):
            torch.cuda._sleep(10 * FIFTY_MIL_CYCLES)
            return self.net(x)

    def __getstate__(self):
        return {}