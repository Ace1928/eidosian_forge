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
def _register_binary_fusion():
    binary_ops = [aten.add, ops.add, aten.sub, ops.sub]
    fusion_ops = [mkldnn._convolution_pointwise.binary, mkldnn._linear_pointwise.binary]
    _computation_user_1 = [_conv_call(users=1), _linear_call(users=1)]
    for computation_call, computation_op, fusion_op in zip(_computation_user_1, computation_ops[:-1], fusion_ops):
        for binary_op in binary_ops:
            pattern = _binary_fusion_v2(computation_call, binary_op)
            _register_binary_unary_fusion_lowering(pattern, computation_op, binary_op, fusion_op)
        for binary_op in [aten.add, ops.add]:
            pattern = _binary_fusion_v1(computation_call, binary_op)
            _register_binary_unary_fusion_lowering(pattern, computation_op, binary_op, fusion_op)