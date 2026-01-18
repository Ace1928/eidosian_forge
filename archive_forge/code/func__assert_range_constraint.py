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
def _assert_range_constraint(self, proxy, lower, upper, assert_msg):
    if lower > -math.inf:
        self._insert_assert_async(operator.ge, proxy, lower, assert_msg)
    if upper < math.inf:
        self._insert_assert_async(operator.le, proxy, upper, assert_msg)