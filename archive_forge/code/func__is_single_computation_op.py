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
def _is_single_computation_op(computation_op):

    def fn(match):
        computation_nodes = filter_nodes(match.nodes, computation_op)
        if len(computation_nodes) < 1:
            return False
        if any((n.args[-3] != 'none' for n in computation_nodes)):
            return False
        return True
    return fn