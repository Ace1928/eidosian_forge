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
def _assert_top_level_events(self, process_global_events, expected_top_level_event_names):
    top_level_event_names = []
    for thread_local_events in process_global_events:
        last_end_time = 0
        for event in thread_local_events:
            event_name = event.name
            time_range = event.time_range
            if time_range.start > last_end_time:
                top_level_event_names.append(event_name)
                last_end_time = time_range.end
    top_level_event_names = sorted(top_level_event_names)
    expected_top_level_event_names = sorted(expected_top_level_event_names)
    self.assertEqual(top_level_event_names, expected_top_level_event_names, f'Expected events {expected_top_level_event_names}, but got {top_level_event_names}')