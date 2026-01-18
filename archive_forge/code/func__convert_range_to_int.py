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
def _convert_range_to_int(range: ValueRanges):
    assert isinstance(range, ValueRanges)
    min_val = _convert_to_int(range.lower)
    max_val = _convert_to_int(range.upper)
    return (min_val, max_val)