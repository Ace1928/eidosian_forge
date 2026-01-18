from collections import deque
import contextlib
import functools
from itertools import chain
import logging
from typing import Any, Callable, Deque, Dict, Generator, List, Optional, Union
import torch
from torch import nn
from torch.autograd import Variable
import torch.autograd.profiler as profiler
import torch.distributed as dist
from fairscale.internal.params import Workhandle, get_global_rank
from fairscale.nn.misc import GradBucket
from fairscale.optim import OSS
def _try_consume_work_handle(self) -> None:
    """Try to consume the oldest future. This is non blocking, if not ready we'll pass"""
    while len(self._work_handles) > 0 and self._work_handles[0].handle.is_completed():
        work_handle = self._work_handles.popleft()
        if work_handle.callback is not None:
            work_handle.callback()