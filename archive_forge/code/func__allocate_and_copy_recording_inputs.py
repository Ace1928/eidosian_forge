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
def _allocate_and_copy_recording_inputs(self, inputs) -> List[Union[torch.Tensor, int]]:
    """
        Allocate inputs for non static, non cudagraph managraphed managed tensors in the memory pool
        and copy over the tensor values.
        """
    torch.cuda.synchronize()
    self.stream.wait_stream(torch.cuda.current_stream())
    recording_inputs: List[Union[Tensor, int]] = []
    with warnings.catch_warnings(record=True), torch.cuda.device(self.device), _use_cuda_memory_pool_manager(self.device, mem_pool=self.cuda_graphs_pool, stream=self.stream):
        for i, inp in enumerate(inputs):
            if not isinstance(inp, torch.Tensor):
                assert isinstance(inp, int)
                recording_inputs.append(inp)
            elif i not in self.static_input_idxs:
                recording_inputs.append(static_input(inp))
                self._copy_input(i, recording_inputs[-1], inp)
                inputs[i] = None
                del inp
            else:
                recording_inputs.append(inp)
    return recording_inputs