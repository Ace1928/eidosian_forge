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
class CUDAGraphTreeManager:
    """
    Groups individual recordings or executions of cuda graphs into a tree of recordings,
    and checks required invariants, and manages warmups of graphs.

    When graphs are recorded in the same tree, it enforces subsequent execution
    to follow the same order and have the same output tensor livespans. To remove
    unnecessary coupling of cuda graphs (and additional imposed invariants),
    the tree manager will end a currently recording tree whenever it is valid - when
    the memory pool no longer has any live allocations.

    We ignore outputs from a previous generation that correspond to prior model outputs.
    Currently this is hardcoded `GenerationTracker.generation` tracked in torch dynamo.
    # TODO: make generation increment configurable, warn on overwrite.

    We run graph warmups in the cudagraph memory pool and return the result on the first invocation
    of a function. For many models it is important to reclaim activations as you run the backward.
    If we were to warm up the model and keep an extra copy of the inputs around to subsequently
    use for recording, we would incur a memory penalty. Additionally, if we are part way through training
    your model and need to recompile, memory will be allocated to the cuda graph pool, so we run this
    warmup run in the cuda graph memory pool. As for recording, warm up needs the state of live tensors
    to be accurately reflected so we checkpoint the allocator state if we need to warm up following graph
    replay.
    """

    def __init__(self, device_index: int):
        self.roots: Dict[FunctionID, List[CUDAGraphNode]] = defaultdict(list)
        self.ids_to_funcs: Dict[FunctionID, WrappedFunction] = {}
        self.ids_to_stack_traces: Dict[FunctionID, StackTraces] = {}
        self.warmed_up_functions: Set[FunctionID] = set()
        self.warned_functions: Set[FunctionID] = set()
        torch._C._set_cached_tensors_enabled(True)
        with torch.cuda.device(device_index):
            torch.cuda.synchronize()
            self.stream = torch.cuda.Stream()
            self.stream.wait_stream(torch.cuda.current_stream())
            self.graph: Optional[torch.cuda.CUDAGraph] = torch.cuda.CUDAGraph()
            self.cuda_graphs_thread_pool = torch.cuda.graph_pool_handle()
            with warnings.catch_warnings(record=True), torch.cuda.graph(self.graph, pool=self.cuda_graphs_thread_pool, stream=self.stream, capture_error_mode='thread_local'):
                pass
        self.graph_counter = itertools.count(0)
        self.func_counter = itertools.count(0)
        self.path_state = ExecutionState.NONE
        self.device_index = device_index
        self.current_node: Optional[CUDAGraphNode] = None
        self.current_gen: int = -1
        self.debug_fail_counter = 0
        self.debug_checkpointing_counter = 0
        self.id_to_mode: Dict[FunctionID, CompilationMode] = {}
        self.running_forwards_with_pending_backwards = False

    def run(self, new_inputs: List[Tensor], function_id: FunctionID):
        assert self.graph is not None, 'Running CUDAGraph after shutdown'
        out = self._run(new_inputs, function_id)
        mode = self.id_to_mode[function_id]
        if mode == CompilationMode.FORWARD:
            self.running_forwards_with_pending_backwards = True
        elif mode == CompilationMode.BACKWARD:
            self.running_forwards_with_pending_backwards = False
        return out

    def set_to_running_backward(self):
        self.running_forwards_with_pending_backwards = False

    def _run(self, new_inputs: List[Tensor], function_id: FunctionID):
        if self.in_recording:
            self.try_end_curr_recording(function_id)
        if self.in_warmup:
            self.try_end_curr_warmup(function_id)
        if not (function_id in self.warmed_up_functions or config.triton.skip_cudagraph_warmup) or self.in_warmup:
            if self.path_state == ExecutionState.EXECUTION:
                self.apply_checkpoint_execution_state_in_allocator()
            return self.run_eager(new_inputs, function_id)
        child_nodes = self.roots if self.current_node is None else self.current_node.children
        if not self.in_recording:
            for child in child_nodes[function_id]:
                if child.check_invariants(new_inputs):
                    return self.execute_node(child, new_inputs)
            if self.current_node is not None and function_id in self.roots:
                self.try_end_curr_execution()
                if self.current_node is None:
                    return self.run(new_inputs, function_id)
            self.debug_fail_counter += 1
            self.try_end_curr_execution()
            if self.current_node is not None:
                self.apply_checkpoint_execution_state_in_allocator()
        return self.record_function(new_inputs, function_id)

    def shutdown(self):
        """
        Remove all cached tensors in all nodes. Because cached tensors can hold gradients which in turn
        might reference a backward which invokes a CUDA Graph Node, we have to manually clear them on shutdown
        to avoid a reference cycle.
        """
        nodes = []
        for roots in self.roots.values():
            nodes.extend(roots)
        while nodes:
            node = nodes.pop()
            for children in node.children.values():
                nodes.extend(children)
            node.remove_node_cached_tensors()
            node.graph = None
        self.graph = None
        self.roots = None
        self.current_node = None

    def record_function(self, new_inputs, function_id) -> List[Optional[Tensor]]:
        graph_id = self.new_graph_id()
        log.debug('Recording function %d of graph recording id %d', function_id.id, graph_id.id)
        torch.cuda.synchronize()
        node = CUDAGraphNode(self.ids_to_funcs[function_id], graph_id, self.current_node, new_inputs, self.cuda_graphs_thread_pool, self.device_index, self.ids_to_stack_traces[function_id], self.stream)
        if self.current_node is None:
            self.roots[function_id].append(node)
        else:
            self.current_node.add_child(function_id, node)
        self.current_node = node
        self.path_state = ExecutionState.RECORDING
        self.update_generation()
        torch.cuda.synchronize()
        return node.run_first_inputs(new_inputs)

    def execute_node(self, node: CUDAGraphNode, new_inputs) -> List[Optional[Tensor]]:
        self.current_node = node
        self.path_state = ExecutionState.EXECUTION
        self.update_generation()
        return node.run(new_inputs)

    def run_eager(self, new_inputs, function_id: FunctionID):
        already_warm = function_id in self.warmed_up_functions
        if not already_warm:
            log.debug('Running warmup of function %d', function_id.id)
        else:
            log.debug('Running eager of function %d because ancestor needed to warm up', function_id.id)
        self.warmed_up_functions.add(function_id)
        node = CUDAWarmupNode(self.ids_to_funcs[function_id], self.current_node, self.cuda_graphs_thread_pool, self.graph, self.device_index, self.ids_to_stack_traces[function_id], self.stream, already_warm)
        self.current_node = node
        self.path_state = ExecutionState.WARMUP
        self.update_generation()
        return node.run(new_inputs)

    def new_graph_id(self) -> GraphID:
        return GraphID(next(self.graph_counter))

    def new_func_id(self) -> FunctionID:
        return FunctionID(next(self.func_counter))

    def add_function(self, model, inputs, static_input_idxs, stack_traces, mode, constants) -> Tuple[Callable[..., Any], List[Optional[Tensor]]]:
        id = self.new_func_id()
        self.ids_to_stack_traces[id] = stack_traces
        self.ids_to_funcs[id] = WrappedFunction(model, static_input_idxs, id, tuple((t for t in constants if isinstance(t, torch.Tensor) and t.is_cuda)))
        self.id_to_mode[id] = mode
        fn = functools.partial(self.run, function_id=id)
        get_container(self.device_index).add_strong_reference(fn)
        return (fn, fn(inputs))

    @property
    def in_recording(self):
        return self.path_state == ExecutionState.RECORDING

    @property
    def in_warmup(self):
        return self.path_state == ExecutionState.WARMUP

    def get_roots(self) -> Iterator[CUDAGraphNode]:
        for nodes in self.roots.values():
            yield from nodes

    @property
    def current_node(self):
        return self._current_node

    @current_node.setter
    def current_node(self, value):
        self._current_node = value
        if value is None:
            self.path_state = ExecutionState.NONE

    def update_generation(self):
        self.current_gen = self.get_curr_generation()

    @staticmethod
    def get_curr_generation() -> int:
        if MarkStepBox.mark_step_counter != 0:
            return MarkStepBox.mark_step_counter
        return GenerationTracker.generation

    @staticmethod
    def user_invoked_mark_step():
        return MarkStepBox.mark_step_counter != 0

    def can_start_new_generation(self) -> bool:
        if not self.in_new_torch_compile_invocation():
            return False
        if self.user_invoked_mark_step():
            return True
        return not self.running_forwards_with_pending_backwards

    def in_new_torch_compile_invocation(self):
        return self.current_gen != self.get_curr_generation()

    def try_end_curr_recording(self, function_id: FunctionID) -> None:
        """
        Check if the current recording can be terminated, either because all outputs of the
        previously recorded node are dead or because it was executed in a different
        generation. Will set current_node to None and in_recording to False if successful.
        """
        assert self.in_recording
        assert self.current_node is not None
        if self.can_start_new_generation():
            self.dealloc_current_path_weakrefs()
            self.clear_current_path_state_and_set_to_none()
            return
        if self.current_node.all_outputs_are_dead():
            self.clear_current_path_state_and_set_to_none()
            return
        self.check_warn_on_unable_to_start_executing(function_id)

    def try_end_curr_execution(self) -> None:
        """
        Check if the current executing node can be terminated, either because all outputs of the
        previously executed node are dead or because it was executed in a different generation.
        Will set current_node to None if successful.
        """
        assert not self.in_recording
        if self.current_node is None:
            return
        if self.can_start_new_generation():
            self.clear_current_path_state_and_set_to_none()
            return
        if self.current_node.all_outputs_are_dead():
            self.clear_current_path_state_and_set_to_none()

    def try_end_curr_warmup(self, function_id: FunctionID):
        if self.can_start_new_generation():
            self.dealloc_current_path_weakrefs()
            self.current_node = None
            return
        if self.current_node.all_outputs_are_dead():
            self.current_node = None
            return
        self.check_warn_on_unable_to_start_executing(function_id)

    def check_warn_on_unable_to_start_executing(self, function_id: FunctionID):
        """Warn if we in a potential loop where we are unable to hit fast path"""
        if function_id in self.warned_functions or not self.in_new_torch_compile_invocation():
            return
        existing_nodes = [node for node in self.current_node._path_from_root if node.wrapped_function.id == function_id]
        if len(existing_nodes) <= 1:
            return
        parents = {n.parent.wrapped_function.id for n in itertools.chain(existing_nodes, (self.current_node,)) if n.parent is not None}
        if len(parents) == len(existing_nodes):
            return
        self.warned_functions.add(function_id)
        warnings.warn('Unable to hit fast path of CUDAGraphs because of pending, uninvoked backwards. Consider running with torch.no_grad() or using torch.compiler.cudagraph_mark_step_begin() before each model invocation')

    def dealloc_current_path_weakrefs(self):
        for node in self.current_node._path_from_root:
            assert len(node.tensor_weakrefs) == len(node.stack_traces)
            for t, stack_trace in zip(node.tensor_weakrefs, node.stack_traces):
                ten = None if t is None else t()
                if ten is None:
                    continue
                stack_trace = stack_trace.strip() if stack_trace else '[Could not find stack trace]'
                msg = f'Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: {stack_trace}. To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.'
                torch._C._set_storage_access_error_msg(ten, msg)
        deleted = set()
        for storage_ref in self.current_node.path_live_weakrefs():
            if storage_ref() and storage_ref.data_ptr() not in deleted:
                deleted.add(storage_ref.data_ptr())
                torch._C._free_And_Remove_DeleterFn(storage_ref())

    def clear_current_path_state_and_set_to_none(self):
        self.current_node.clear_path_state()
        self.current_node = None

    def apply_checkpoint_execution_state_in_allocator(self):
        """
        Checkpoint the current execution state in the caching allocator so that
        additional cudagraph recordings can be made respecting existent live storages.
        """
        self.debug_checkpointing_counter += 1
        log.debug('Checkpointing cuda caching allocator state. Number of checkpoints %d', self.debug_checkpointing_counter)
        state = self.current_node.checkpointed_caching_state
        device = self.current_node.device
        assert state is not None and device is not None
        stale_storages: List[int] = []
        self.current_node.remove_path_cached_tensors()
        live_storages_wrappers = list(self.current_node.path_live_weakrefs())
        live_storages_weak_refs = [t() for t in live_storages_wrappers]
        ptrs_to_deallocate = self.current_node.data_ptrs_dead_since_invocation()
        torch._C._cuda_setCheckpointPoolState(device, state, stale_storages, live_storages_weak_refs)
        for ptr in set(ptrs_to_deallocate):
            torch._C._cuda_cudaCachingAllocator_raw_delete(ptr)
        if config.triton.slow_path_cudagraph_asserts:
            check_memory_pool(self.device_index, self.cuda_graphs_thread_pool, live_storages_wrappers)
            for wrapper in live_storages_wrappers:
                assert wrapper()
                assert torch._C._has_Standard_Deleter(wrapper())
                assert wrapper.data_ptr() not in ptrs_to_deallocate

    def live_cudagraph_pool_storages_in_curr_execution(self) -> List[StorageWeakRefPointer]:
        if self.current_node is None:
            return []
        return [t() for t in self.current_node.path_live_weakrefs()]