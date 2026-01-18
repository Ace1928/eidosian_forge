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
def _memory_profile(self) -> MemoryProfile:
    required = ('record_shapes', 'profile_memory', 'with_stack')
    missing = [f'{i}=True' for i in required if not getattr(self, i)]
    if missing:
        raise ValueError(f'{', '.join(missing)} required for memory profiling.')
    assert self.profiler is not None and self.profiler.kineto_results is not None
    return MemoryProfile(self.profiler.kineto_results)