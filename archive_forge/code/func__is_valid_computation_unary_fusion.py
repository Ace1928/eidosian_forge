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
def _is_valid_computation_unary_fusion(computation_op, is_bf16=False):

    def fn(match):
        matched = _is_single_computation_op(computation_op)(match)
        computation_node = filter_nodes(match.nodes, computation_op)[0]
        if is_bf16:
            conversion_dtype_nodes = filter_nodes(match.nodes, prims.convert_element_type.default)
            if len(conversion_dtype_nodes) != 2:
                return False
            if computation_node == conversion_dtype_nodes[0].args[0]:
                to_float = conversion_dtype_nodes[0].args[1]
                to_bf16 = conversion_dtype_nodes[1].args[1]
            else:
                to_float = conversion_dtype_nodes[1].args[1]
                to_bf16 = conversion_dtype_nodes[0].args[1]
            matched = matched and to_float == torch.float and (to_bf16 == torch.bfloat16)
        return matched
    return fn