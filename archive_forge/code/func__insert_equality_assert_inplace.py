import copy
import math
import operator
import traceback
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Set, Tuple
import sympy
import torch
import torch.fx
from torch.fx.experimental.symbolic_shapes import SymInt
from torch._export.pass_base import _ExportPassBase, ProxyValue, PassResult
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._sympy.value_ranges import ValueRanges
def _insert_equality_assert_inplace(self, graph: torch.fx.Graph, inputdim_to_node: Dict[InputDim, torch.fx.Node]):
    for input_dim, other_input_dim in self.equality_constraints:
        dim_node = inputdim_to_node[input_dim]
        assert_msg = f'Input {input_dim.input_name}.shape[{input_dim.dim}] is not equal to input {other_input_dim.input_name}.shape[{other_input_dim.dim}]'
        other_dim_node = inputdim_to_node[other_input_dim]
        self._insert_assert_async_inplace(graph, operator.eq, (dim_node, other_dim_node), assert_msg)