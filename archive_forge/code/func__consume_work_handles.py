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
def _consume_work_handles(self) -> None:
    """Consume all the futures which are tied to this optimizer's buckets.
        We start from the first/older ones, since they are the most likely to be ready and non-blocking
        """
    while len(self._work_handles) > 0:
        work_handle = self._work_handles.popleft()
        work_handle.handle.wait()
        if work_handle.callback is not None:
            work_handle.callback()