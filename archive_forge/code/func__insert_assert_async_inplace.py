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
def _insert_assert_async_inplace(self, graph, operator, args, assert_msg):
    """
        Inserts assert_async call_function nodes in the graph. This function is
        called before we run the interpreter-based pass and does an inplace
        insertion.
        """
    cmp_node = graph.call_function(operator, args)
    with graph.inserting_after(cmp_node):
        cmp_tensor_node = graph.call_function(torch.ops.aten.scalar_tensor.default, (cmp_node,))
    with graph.inserting_after(cmp_tensor_node):
        _ = graph.call_function(torch.ops.aten._assert_async.msg, (cmp_tensor_node, assert_msg))