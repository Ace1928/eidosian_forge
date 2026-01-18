import gzip
import json
import os
import tempfile
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from warnings import warn
import torch
import torch.autograd.profiler as prof
from torch._C import _get_privateuse1_backend_name
from torch._C._profiler import (
from torch.autograd import kineto_available, ProfilerActivity
from torch.profiler._memory_profiler import MemoryProfile, MemoryProfileTimeline
def _get_distributed_info(self):
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized():
        return None
    return {'backend': dist.get_backend(), 'rank': dist.get_rank(), 'world_size': dist.get_world_size()}