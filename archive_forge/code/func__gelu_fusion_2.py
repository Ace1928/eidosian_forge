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
def _gelu_fusion_2(computation_call):
    return CallFunction(aten.mul, CallFunction(aten.mul, computation_call, 0.5), CallFunction(aten.add, CallFunction(aten.tanh, CallFunction(aten.mul, CallFunction(aten.add, computation_call, CallFunction(aten.mul, CallFunction(aten.mul, CallFunction(aten.mul, computation_call, computation_call), computation_call), 0.044715)), 0.7978845608028654)), 1))