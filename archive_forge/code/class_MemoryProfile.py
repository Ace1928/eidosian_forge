import collections
import dataclasses
import enum
import itertools as it
import logging
from typing import (
from typing_extensions import Literal
import torch
from torch._C import FunctionSchema
from torch._C._autograd import _ProfilerResult
from torch._C._profiler import (
from torch._utils import _element_size
from torch.profiler import _utils
class MemoryProfile:

    def __init__(self, result: _ProfilerResult) -> None:
        self._op_tree = OpTree(result)
        self._data_flow_graph = DataFlowGraph(self._op_tree)
        self._size_map = SizeMap(self._op_tree)
        self._categories = CategoryDict()
        self._set_gradients_and_temporaries()
        self._set_parameters_using_python_tracer()
        self._set_inputs()
        self._set_parameters_using_data_flow()
        self._set_activations()
        self._set_optimizer_state()
        self._set_autograd_detail()

    @property
    def timeline(self) -> Tuple[Tuple[int, Action, KeyAndID, int], ...]:
        output: List[Tuple[int, Action, KeyAndID, int]] = []
        allocation_times: Dict[Tuple[TensorKey, bool], int] = {}
        live_unknown: Dict[Tuple[int, torch.device], Literal[True]] = {}
        for event in self._op_tree.dfs():
            if event.typed[0] == _EventType.Allocation:
                alloc_fields = event.typed[1]
                alloc_size = alloc_fields.alloc_size
                is_allocation = alloc_size > 0
                t = event.start_time_ns
                tkey = TensorKey.from_allocation(alloc_fields)
                if tkey is not None:
                    allocation_times[tkey, is_allocation] = t
                else:
                    key = Key(alloc_fields.device)
                    ptr_and_device = (alloc_fields.ptr, key.device)
                    if is_allocation:
                        if ptr_and_device in live_unknown:
                            output.append((t, Action.INCREMENT_VERSION, (key, 0), alloc_size))
                        else:
                            live_unknown[ptr_and_device] = True
                            output.append((t, Action.CREATE, (key, 0), alloc_size))
                    else:
                        output.append((t, Action.DESTROY, (key, 0), -alloc_size))
                        if not live_unknown.pop(ptr_and_device, False):
                            output.append((-1, Action.PREEXISTING, (key, 0), -alloc_size))
        snapshot = self._category_snapshot()
        last_version = dict(sorted(snapshot.keys()))
        events: List[Tuple[int, Action, TensorAndID]] = [(-1, Action.PREEXISTING, (key, version)) for key, version in snapshot.keys() if (key, True) not in allocation_times and version == 0]
        for node in self._data_flow_graph.flow_nodes:
            for key, edge in node._edges.items():
                if edge.is_allocation:
                    t = allocation_times[key, True]
                    events.append((t, Action.CREATE, (key, 0)))
                elif edge.mutated:
                    t = node._event.start_time_ns
                    version = edge.input_version
                    assert version is not None
                    events.append((t, Action.INCREMENT_VERSION, (key, version)))
                if edge.is_deletion:
                    t = allocation_times[key, False]
                    events.append((t, Action.DESTROY, (key, last_version[key])))
        output.extend(((time, action, (key, version), self._size_map[key]) for time, action, (key, version) in events))
        output.sort(key=lambda x: (x[0], x[1].value))
        return tuple(output)

    def _is_gradient(self, *args, **kwargs) -> bool:
        return self._categories.get(*args, **kwargs) == Category.GRADIENT

    def _category_snapshot(self) -> Dict[TensorAndID, Optional[Category]]:
        all_tensor_versions: Set[TensorAndID] = set()
        for node in self._data_flow_graph.flow_nodes:
            all_tensor_versions.update(((k, v) for k, (_, v) in node.inputs.items()))
            all_tensor_versions.update(((key, 0) for key in node.intermediates))
            all_tensor_versions.update(node.outputs.items())
        for i in self._categories._values.values():
            all_tensor_versions.update(((key, 0) for key in i._by_id_keyset))
        return {(key, version): self._categories.get(key, version) for key, version in sorted(all_tensor_versions)}

    def _any_version_depends_on_gradient(self) -> Set[int]:
        """Extract IDs of Tensors which depend or will depend on a gradient.

        Note that this weakened definition of "depends" requires us to loop
        over the data flow graph multiple times because it allows dependency
        information to flow backward through edges and removes the guarantee
        that nodes are topologically sorted. (Or indeed, even that a valid
        topological order exists.) Put another way, we have converted an
        acyclic data flow graph into a cyclic graph and we are attempting to
        partition cycles involving a gradient from the rest of the graph.
        """
        depends_on_gradient: Set[int] = set()
        while True:
            start_size = len(depends_on_gradient)
            for node in self._data_flow_graph.flow_nodes:
                ids = tuple((key.id for key, (_, version) in node.inputs.items() if self._categories.get(key, version) in (Category.GRADIENT, Category.PARAMETER) or key.id in depends_on_gradient))
                if ids:
                    depends_on_gradient.update(ids)
                    depends_on_gradient.update((key.id for key in node.outputs))
            if len(depends_on_gradient) == start_size:
                return depends_on_gradient

    def _set_gradients_and_temporaries(self) -> None:
        """Mark Tensors which are unambiguous and simple to reason about."""
        for event in self._op_tree.dfs():
            for _, p_grad in extract_gradients(event):
                self._categories.set_by_id(p_grad, Category.GRADIENT)
        for node in self._data_flow_graph.flow_nodes:
            for i in node.intermediates:
                self._categories.set_by_key(i, Category.TEMPORARY)

    def _set_parameters_using_python_tracer(self) -> None:
        for event in self._op_tree.dfs():
            for p in extract_parameters(event):
                if p is not None:
                    self._categories.set_by_id(p, Category.PARAMETER)

    def _set_inputs(self) -> None:
        """Mark inputs based on which Tensors are updated using gradients.

        The process for differentiating between inputs and activations is more
        involved. Most Tensors in a training loop depend on at least one
        gradient: parameters depend on them through updates, and activations
        and optimizer state depend on them transitively through parameters.
        Critically, we do not need to know which Tensors are parameters to
        apply this method; we can simply walk the data flow graph to build the
        set of all values which depend on a gradient and then obtain the set
        of inputs from the conjugate set.

        There is, however, one hiccup. The first time we see a parameter is
        generally on the forward pass of the first step. We know from
        inspection of the data flow graph that v1 of that Tensor depends on
        a gradient (provided we profile an optimizer step), but not v0. To
        address this problem we weaken the definition of "depends on a
        gradient" to "any version of this Tensor depends on a gradient",
        which in turn strengthens the criteria for the input set enough to
        filter the activations in the forward pass of the first step."""
        depends_on_gradient = self._any_version_depends_on_gradient()
        produces_gradient: Set[TensorAndID] = set()
        for node in reversed(self._data_flow_graph.flow_nodes):
            tensors = {(key, version) for key, (_, version) in node.inputs.items()}
            tensors |= node.outputs.items()
            if any((self._categories.get(*i) in (Category.GRADIENT, Category.PARAMETER) or i in produces_gradient for i in tensors)):
                produces_gradient |= tensors
        input_candidates = produces_gradient.copy()
        for node in self._data_flow_graph.flow_nodes:
            if RecordScope.BACKWARD_FUNCTION in get_scopes(node._event):
                input_candidates -= set(node.outputs.items())
        for key, version in input_candidates:
            if key.id not in depends_on_gradient:
                self._categories.setdefault_by_version(key, version, Category.INPUT)

    def _set_parameters_using_data_flow(self) -> None:
        """Deduce which Tensors are parameters.

        Consider the following code for the step of SGD with momentum
        (nesterov=False), where `d_p` is the gradient of `param` and `buf` is
        the momentum buffer.
        ```
          buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
          d_p = buf
          param.add_(d_p, alpha=-lr)
        ```
        Both `param` and `buf` take a gradient and perform an in-place update.

        The python tracer will inspect calls to `nn.Module.forward` and
        `optim.Optimizer.step` to extract parameter and optimizer state
        respectively (including parameters), so this is generally a non-issue.

        However as a fallback we can also exploit several properties of
        parameters to distinguish them from other model state.

        First, they are directly used in the forward pass. (At this point we
        haven't established which parts of the graph correspond to the forward
        pass but we can deduce enough to suffice.) Some mutable state such as
        batch norm moving averages also contribute to the forward pass, but
        optimizer state does not.

        Second, a parameter is by definition used to compute at least one
        gradient and depends on at least one gradient.
        """
        snapshot = self._category_snapshot()
        candidate_parameters: Set[TensorAndID] = set()
        candidate_fwd_tensors: Set[TensorAndID] = {i for i, category in snapshot.items() if category == Category.INPUT}
        for node in self._data_flow_graph.flow_nodes:
            inputs = {(key, value) for key, (_, value) in node.inputs.items()}
            if RecordScope.BACKWARD_FUNCTION not in get_scopes(node._event) and (not any((self._is_gradient(*i) for i in inputs))) and (not any((self._is_gradient(*i) for i in node.outputs.items()))) and candidate_fwd_tensors.intersection(inputs):
                candidate_fwd_tensors |= node.outputs.items()
                candidate_parameters |= inputs.difference(candidate_fwd_tensors)
        used_for_gradient: Set[TensorAndID] = set()
        for node in reversed(self._data_flow_graph.flow_nodes):
            if any((self._is_gradient(*i) or i in used_for_gradient for i in node.outputs.items())):
                for key, (_, version) in node.inputs.items():
                    used_for_gradient.add((key, version))
        candidate_parameters.intersection_update(used_for_gradient)
        parameter_keys = {key.id for key, _ in candidate_parameters}
        parameter_keys &= self._any_version_depends_on_gradient()
        for key, _ in snapshot.keys():
            if key.id in parameter_keys:
                self._categories.set_by_id(key, Category.PARAMETER)

    def _set_activations(self) -> None:
        """Flood the graph to identify activations."""
        required = {Category.INPUT, Category.ACTIVATION}
        also_allowed = {Category.PARAMETER, Category.TEMPORARY}
        for node in self._data_flow_graph.flow_nodes:
            inputs = {(key, value) for key, (_, value) in node.inputs.items()}
            input_categories = {self._categories.get(*i) for i in inputs}
            if input_categories & required and (not input_categories - (required | also_allowed)) and (RecordScope.BACKWARD_FUNCTION not in get_scopes(node._event)):
                for i in node.outputs.items():
                    self._categories.setdefault_by_version(*i, Category.ACTIVATION)

    def _set_optimizer_state(self) -> None:
        for event in self._op_tree.dfs():
            if event.typed[0] == _EventType.PyCall and event.typed[1].optimizer:
                parameters = event.typed[1].optimizer.parameters
                for _, t in it.chain(*[state for _, _, state in parameters]):
                    key = TensorKey.from_tensor(t)
                    if key is not None:
                        self._categories.set_by_id(key, Category.OPTIMIZER_STATE)

    def _set_autograd_detail(self):
        prior = {None, Category.AUTOGRAD_DETAIL}
        for node in self._data_flow_graph.flow_nodes:
            if RecordScope.BACKWARD_FUNCTION in get_scopes(node._event):
                for key, version in node.outputs.items():
                    if version == 0 or self._categories.get(key, version - 1) in prior:
                        self._categories.setdefault_by_version(key, version, Category.AUTOGRAD_DETAIL)