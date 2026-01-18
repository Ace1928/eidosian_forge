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
@staticmethod
def _get_different_indices(prev: List[List[bool]], curr: List[List[bool]]) -> List[PathOutputIndex]:
    """Find indices where the two lists differ."""
    dead_indices = []
    assert len(prev) <= len(curr)
    for i, (outputs1, outputs2) in enumerate(zip(prev, curr)):
        assert len(outputs1) == len(outputs2)
        for j, (output1, output2) in enumerate(zip(outputs1, outputs2)):
            if output1 != output2:
                dead_indices.append((i, j))
    return dead_indices