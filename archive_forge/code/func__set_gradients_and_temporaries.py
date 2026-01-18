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
def _set_gradients_and_temporaries(self) -> None:
    """Mark Tensors which are unambiguous and simple to reason about."""
    for event in self._op_tree.dfs():
        for _, p_grad in extract_gradients(event):
            self._categories.set_by_id(p_grad, Category.GRADIENT)
    for node in self._data_flow_graph.flow_nodes:
        for i in node.intermediates:
            self._categories.set_by_key(i, Category.TEMPORARY)