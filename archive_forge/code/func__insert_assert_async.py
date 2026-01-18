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
def _insert_assert_async(self, operator, lower, upper, assert_msg):
    """
        Inserts assert_async call_function nodes in the graph. This function is
        called **during** the interpreter-based pass.
        """
    self.counter += 1
    cmp = super().call_operator(operator, (lower, upper), {}, self._create_dummy_node_metadata())
    cmp_tensor = super().call_operator(torch.ops.aten.scalar_tensor.default, (cmp,), {}, self._create_dummy_node_metadata())
    super().call_operator(torch.ops.aten._assert_async.msg, (cmp_tensor, assert_msg), {}, self._create_dummy_node_metadata())