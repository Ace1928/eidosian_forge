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
def _run_test_profiler_remote_events_profiled(self):
    if self.rank != 1:
        return
    dst_ranks = [rank for rank in range(0, self.world_size) if rank != self.rank]
    for dst in dst_ranks:
        dst_worker = worker_name(dst)
        with _profile() as prof:
            fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
            ret = fut.wait()
        events = prof.function_events
        rpc_event = get_function_event(events, RPCExecMode.ASYNC.value)
        self.check_profiling_info(worker_name(self.rank), dst_worker, udf_with_torch_ops, rpc_event, RPCExecMode.ASYNC)
        remote_events = {event.name: event for event in events if event.is_remote}
        rpc_profiling_key = _build_rpc_profiling_key(RPCExecMode.ASYNC, udf_with_torch_ops.__qualname__, worker_name(self.rank), worker_name(dst))
        for expected_remote_event_name in EXPECTED_REMOTE_EVENTS:
            expected_key = rpc_profiling_key + REMOTE_OP_STR + expected_remote_event_name
            self.assertTrue(expected_key in remote_events)
            remote_event = remote_events[expected_key]
            self.assertEqual(remote_event.node_id, dst)

        def convert_remote_to_local(event_name):
            remote_op_key = rpc_profiling_key + REMOTE_OP_STR
            return event_name[event_name.find(remote_op_key) + len(remote_op_key):]
        remote_events_list = [convert_remote_to_local(event.name) for event in events if convert_remote_to_local(event.name) in EXPECTED_REMOTE_EVENTS]
        self.assertEqual(set(remote_events_list), set(EXPECTED_REMOTE_EVENTS), f'Mismatch between profiled events: {set(remote_events_list)} and expected events: {set(EXPECTED_REMOTE_EVENTS)}')