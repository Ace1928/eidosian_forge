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
def _insert_specialized_shape_assert_inplace(self, graph: torch.fx.Graph, input_dim: InputDim, dim_node: torch.fx.Node, shape: int):
    assert_msg = f'Input {input_dim.input_name}.shape[{input_dim.dim}] is specialized at {shape}'
    with graph.inserting_after(dim_node):
        eq_node = graph.call_function(operator.eq, (dim_node, shape))
    with graph.inserting_after(eq_node):
        tensor_eq_node = graph.call_function(torch.ops.aten.scalar_tensor.default, (eq_node,))
    with graph.inserting_after(tensor_eq_node):
        _ = graph.call_function(torch.ops.aten._assert_async.msg, (tensor_eq_node, assert_msg))