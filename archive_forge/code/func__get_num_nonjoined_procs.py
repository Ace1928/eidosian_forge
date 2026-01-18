import warnings
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, List, NamedTuple, Optional, Type
import torch
import torch.distributed as dist
def _get_num_nonjoined_procs(self):
    """Return the number of non-joined processes by shadowing an all-reduce in the non-joined processes."""
    num_nonjoined_procs = torch.zeros(1, device=self._device)
    dist.all_reduce(num_nonjoined_procs, group=self._process_group)
    return num_nonjoined_procs.item()