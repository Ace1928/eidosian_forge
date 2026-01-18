import torch
from torch.utils._pytree import tree_map
from typing import Iterator, List, Optional
import logging
import contextlib
import itertools
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.weak import WeakTensorKeyDictionary
import functools
from torch._C._profiler import gather_traceback, symbolize_tracebacks
def _shortid(self, t: torch.Tensor) -> int:
    if t not in self.memo:
        self.memo[t] = self.next_id
        self.next_id += 1
    return self.memo[t]