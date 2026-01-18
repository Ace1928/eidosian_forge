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
def _is_valid_computation_binary(computation_op, binary_op, other_index=None):

    def fn(match):
        if not _is_single_computation_op(computation_op)(match):
            return False
        if not _is_valid_binary(match, binary_op):
            return False
        return True
    return fn