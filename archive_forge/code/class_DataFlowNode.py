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
class DataFlowNode:

    def __init__(self, event: _ProfilerEvent, graph: 'DataFlowGraph') -> None:
        self._event = event
        self._graph = graph
        self._edges: Dict[TensorKey, DataFlowEdge] = self._determine_edges()
        for key, edge in self._edges.items():
            if edge.mutated and (not edge.is_allocation):
                self._graph.bump(key)
        versions = {k: (v, self._graph.lookup(k)) for k, v in self.outputs.items()}
        assert all((i == j for i, j in versions.values())), f'{versions}, {self._edges}'

    def _determine_edges(self) -> Dict[TensorKey, DataFlowEdge]:
        subtree = tuple(_utils.traverse_dfs([self._event]))
        mutable_by_key: Dict[Optional[TensorKey], Set[Optional[bool]]] = {}
        for op in (i.typed[1] for i in subtree if i.typed[0] == _EventType.TorchOp):
            for op_input, mutable in zip(op.inputs, SchemaMatcher.inputs_are_mutable(op)):
                if isinstance(op_input, _TensorMetadata):
                    key = TensorKey.from_tensor(op_input)
                    mutable_by_key.setdefault(key, set()).add(mutable)
                elif isinstance(op_input, list):
                    for op_input_i in op_input:
                        key = TensorKey.from_tensor(op_input_i)
                        mutable_by_key.setdefault(key, set()).add(mutable)
        edges: DefaultDict[Optional[TensorKey], DataFlowEdge]
        edges = collections.defaultdict(DataFlowEdge)
        for key, mutable_set in mutable_by_key.items():
            if key is not None:
                edges[key].input_version = self._graph.lookup(key) if key else -1
                mutated = True in mutable_set or tuple(mutable_set) == (None,)
                edges[key].mutated = mutated
        for i in subtree:
            if i.typed[0] == _EventType.Allocation and i.typed[1].alloc_size < 0:
                key = TensorKey.from_allocation(i.typed[1])
                edge = edges[key]
                assert key is None or edge.mutated is not None, f'Double delete: {key}'
                edge.mutated = None
                edge.input_version = self._graph.lookup(key) if key else -1
        for i in subtree:
            if i.typed[0] == _EventType.Allocation and i.typed[1].alloc_size > 0:
                edges[TensorKey.from_allocation(i.typed[1])].input_version = None
        return dict(sorted(((k, v) for k, v in edges.items() if k is not None)))

    @property
    def inputs(self) -> Dict[TensorKey, Tuple[bool, int]]:
        return {k: (bool(v.mutated), cast(int, v.input_version)) for k, v in self._edges.items() if not v.is_allocation}

    @property
    def outputs(self) -> Dict[TensorKey, int]:
        return {k: 0 if v.input_version is None else v.input_version + 1 for k, v in self._edges.items() if v.is_allocation and (not v.is_deletion) or v.mutated}

    @property
    def intermediates(self) -> Tuple[TensorKey, ...]:
        return tuple((k for k, v in self._edges.items() if v.is_allocation and v.is_deletion))

    @property
    def start_time(self) -> int:
        return self._event.start_time_ns