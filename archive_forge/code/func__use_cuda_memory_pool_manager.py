important optimization when chaining multiple CUDA graphs together, as it
from __future__ import annotations
import contextlib
import dataclasses
import functools
import gc
import itertools
import logging
import operator
import sys
import threading
import traceback
import warnings
import weakref
from collections import defaultdict
from enum import auto, Enum
from typing import (
import torch.fx
from torch import Tensor
from torch._dynamo.mutation_guard import GenerationTracker
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.compile_fx import (
from torch.multiprocessing.reductions import StorageWeakRef
from torch.storage import UntypedStorage
from torch.types import _bool
from torch.utils import _pytree as pytree
from torch.utils.weak import TensorWeakRef
from . import config
@contextlib.contextmanager
def _use_cuda_memory_pool_manager(device, mem_pool, stream):
    """
    Context manager to use cuda graph pool for new allocations. If you use this manager
    all cudagraph tensors in use should be reflected in the allocator or they will be overwritten.
    existing_graph should already have been used in a capture, and the mem_pool must already exist,
    because this manager will not preserve a reference to the pool which keeps it alive.
    """
    torch.cuda.synchronize()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.device(device):
        torch._C._cuda_beginAllocateCurrentStreamToPool(device, mem_pool)
        try:
            yield
        finally:
            torch._C._cuda_endAllocateCurrentStreamToPool(device)
            torch._C._cuda_releasePool(device, mem_pool)