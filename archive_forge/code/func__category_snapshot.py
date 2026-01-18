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
def _category_snapshot(self) -> Dict[TensorAndID, Optional[Category]]:
    all_tensor_versions: Set[TensorAndID] = set()
    for node in self._data_flow_graph.flow_nodes:
        all_tensor_versions.update(((k, v) for k, (_, v) in node.inputs.items()))
        all_tensor_versions.update(((key, 0) for key in node.intermediates))
        all_tensor_versions.update(node.outputs.items())
    for i in self._categories._values.values():
        all_tensor_versions.update(((key, 0) for key in i._by_id_keyset))
    return {(key, version): self._categories.get(key, version) for key, version in sorted(all_tensor_versions)}