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
@torch.no_grad()
def _clear_counters(self) -> None:
    """Reset all the grad reduce and call counters"""
    if self.training:
        self._grad_to_be_reduced = [True for _ in self._trainable_params]
    self._bucket_flush_callback_set = False
    if self._use_buckets:
        for bucket in self._bucket_list:
            bucket.reset_checked_in()
    if not self._should_accumulate_grads:
        self._accumulate_grads_flipped = False