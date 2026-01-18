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
def _flush_reduce_calls(self) -> None:
    for bucket in self._bucket_list:
        if not bucket.sent:
            assert bucket.buffer is not None
            bucket.buffer.mul_(self._world_size_scaling)
            self._work_handles.append(Workhandle(handle=dist.reduce(tensor=bucket.buffer, dst=bucket.destination, group=self._process_group, async_op=True), callback=None))
            bucket.sent = True
    self._consume_work_handles()