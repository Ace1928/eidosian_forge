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
class SizeMap:

    def __init__(self, op_tree: OpTree) -> None:
        self._values: Dict[TensorKey, int] = {}
        for node in op_tree.sorted_nodes:
            if node.typed[0] == _EventType.TorchOp:
                for t in self._flat_tensor_inputs(node.typed[1]):
                    self._update_values(t)
            elif node.typed[0] == _EventType.PyCall:
                typed_fields = node.typed[1]
                assert typed_fields.module is None or typed_fields.optimizer is None
                if typed_fields.module is not None:
                    for _, p, p_grad in typed_fields.module.parameters:
                        self._update_values(p)
                        self._update_values(p_grad)
                if typed_fields.optimizer is not None:
                    for p, p_grad, state in typed_fields.optimizer.parameters:
                        self._update_values(p)
                        self._update_values(p_grad)
                        for _, t in state:
                            self._update_values(t)
        allocations: Dict[TensorKey, int] = {}
        for node in op_tree.sorted_nodes:
            if node.typed[0] == _EventType.Allocation:
                alloc_fields = node.typed[1]
                key = TensorKey.from_allocation(alloc_fields)
                if key:
                    new_size = abs(alloc_fields.alloc_size)
                    prior_size = allocations.setdefault(key, new_size)
                    if prior_size != new_size:
                        delta = f'{prior_size} vs. {new_size}'
                        log.warning('Mismatch between allocation and free: %s', delta)
        self._values.update(allocations)

    def _update_values(self, t: Optional[_TensorMetadata]) -> None:
        key = TensorKey.from_tensor(t)
        if key is not None and t is not None and (t.layout == torch.strided):
            n = max((i[0] * i[1] for i in zip(t.sizes or [1], t.strides or [1])))
            num_bytes = n * _element_size(t.dtype)
            assert num_bytes >= 0, f'{num_bytes}'
            self._values[key] = max(self._values.get(key, 0), num_bytes)

    @staticmethod
    def _flat_tensor_inputs(op: _ExtraFields_TorchOp) -> Iterator[_TensorMetadata]:
        for i in op.inputs:
            if isinstance(i, _TensorMetadata):
                yield i
            elif isinstance(i, list):
                yield from i

    def __getitem__(self, key: TensorKey):
        return self._values[key]