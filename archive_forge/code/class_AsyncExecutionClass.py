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
class AsyncExecutionClass:

    @staticmethod
    @rpc.functions.async_execution
    def static_async_add(to, x, y, z):
        return rpc.rpc_async(to, torch.add, args=(x, y)).then(lambda fut: fut.wait() + z)

    @classmethod
    @rpc.functions.async_execution
    def class_async_add(cls, to, x, y, z):
        ret_fut = torch.futures.Future()
        rpc.rpc_async(to, torch.add, args=(x, y)).then(lambda fut: ret_fut.set_result(fut.wait() + z))
        return ret_fut

    @rpc.functions.async_execution
    def bound_async_add(self, to, x, y, z):
        return rpc.rpc_async(to, torch.add, args=(x, y)).then(lambda fut: fut.wait() + z)