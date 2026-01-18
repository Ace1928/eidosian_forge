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
def _add_first_outputs(self, outputs, static_input_persistent_storage_ptrs: Dict[int, StorageWeakRefWrapper]):
    """Add the outputs from the first invocation of the node and set up metadata"""
    prev_liveness = self.recorded_liveness_before_graph
    curr_liveness = self._get_liveness(self.path_weakrefs)
    delta = self._get_different_indices(prev_liveness, curr_liveness)
    self.expected_dead_indices_after_graph = delta
    assert len(self.outputs_weakrefs) == 0
    output_new_storages_index: Dict[StorageDataPtr, int] = {}
    self.unaliased_in_all_paths = [False for _ in range(len(outputs))]
    self.static_output_tensors = [None for _ in range(len(outputs))]
    for i, o in enumerate(outputs):
        if o is None or not isinstance(o, torch.Tensor):
            self.output_storage_alias.append(UnaliasedStorage)
            continue
        (torch._check(o.is_cuda or o.untyped_storage().data_ptr() == 0, lambda: f'Expected all cuda outputs in cuda graph recording. Non cuda output from {(self.stack_traces[i] if self.stack_traces else '(unknown)')}'),)
        ref = static_input_persistent_storage_ptrs.get(o.untyped_storage().data_ptr(), None)
        is_empty_storage = o.untyped_storage().data_ptr() == 0
        if ref and ref() is not None or is_empty_storage:
            self.output_storage_alias.append(None)
            self.static_output_tensors[i] = o
            continue
        path_ref = self._is_alias_of_live_recorded_tensor(o)
        if path_ref is not None:
            self._mark_prior_graph_output_as_aliased(path_ref)
            self.output_storage_alias.append(AliasesPriorGraphOutput(path_ref))
            continue
        if o.untyped_storage().data_ptr() in output_new_storages_index:
            index = output_new_storages_index[o.untyped_storage().data_ptr()]
            self.unaliased_in_all_paths[index] = False
            self.output_storage_alias.append(AliasesNewOutput(index))
            continue
        output_new_storages_index[o.untyped_storage().data_ptr()] = i
        self.output_storage_alias.append(UnaliasedStorage)
        self.unaliased_in_all_paths[i] = True
    if self.stack_traces is None:
        self.stack_traces = [None for _ in range(len(outputs))]
    else:
        assert len(self.stack_traces) == len(outputs), 'Wrong number of stack traces passed in'
    assert not self.outputs_weakrefs
    for out, static_output_tensor in zip(outputs, self.static_output_tensors):
        if not isinstance(out, torch.Tensor) or static_output_tensor is not None:
            self.outputs_weakrefs.append(None)
            self.tensor_weakrefs.append(None)
        else:
            self.outputs_weakrefs.append(StorageWeakRefWrapper(out))
            self.tensor_weakrefs.append(TensorWeakRef(out))
    self.recorded_liveness_after_graph = self._get_liveness(self.path_weakrefs)
    self.checkpointed_caching_state = torch._C._cuda_getCheckpointState(self.device, self.cuda_graphs_pool)
    for depth in range(len(self.path_weakrefs)):
        for output_index in range(len(self.path_weakrefs[depth])):
            if is_live(self.path_weakrefs[depth][output_index]):
                self.live_indices_after_graph.append((depth, output_index))
    self.debug_check_invariants_after_invocation()
    if config.triton.slow_path_cudagraph_asserts:
        check_memory_pool(self.device, self.cuda_graphs_pool, list(self.path_live_weakrefs()))