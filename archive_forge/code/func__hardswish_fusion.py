import functools
import operator
from functools import reduce
from typing import Any, Tuple
import torch
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from .. import ir
from ..lowering import lowerings as L
from ..pattern_matcher import (
from ..virtualized import ops
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern
from .quantization import (
def _hardswish_fusion(computation_call):
    return CallFunction(aten.div, CallFunction(aten.mul, computation_call, CallFunction(aten.clamp_max, CallFunction(aten.clamp_min, CallFunction(aten.add, computation_call, 3), 0), 6)), 6)