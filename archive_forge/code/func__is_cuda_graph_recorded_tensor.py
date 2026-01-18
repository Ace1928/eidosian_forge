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
def _is_cuda_graph_recorded_tensor(self, t: torch.Tensor):
    """Is this tensor an output of a node in this path"""
    for output_refs in self.path_weakrefs:
        for storage_weak_ref in output_refs:
            if storage_weak_ref is None:
                continue
            data_ptr = storage_weak_ref.data_ptr()
            if t.untyped_storage().data_ptr() == data_ptr:
                return True
    return False