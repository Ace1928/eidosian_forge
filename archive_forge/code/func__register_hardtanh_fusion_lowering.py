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
def _register_hardtanh_fusion_lowering(pattern, computation_op, is_bf16=False):

    @register_lowering_pattern(pattern, extra_check=_is_single_computation_op(computation_op))
    def fn(match, *args, **kwargs):
        min_value = kwargs.get('min_value')
        max_value = kwargs.get('max_value')
        if isinstance(min_value, ir.TensorBox) or isinstance(max_value, ir.TensorBox):
            matched = False
        else:
            assert max_value is not None
            matched = min_value <= max_value
        if is_bf16:
            dtype1 = kwargs.get('to_float')
            dtype2 = kwargs.get('to_bf16')
            matched = matched and dtype1 == torch.float and (dtype2 == torch.bfloat16)
        computation_args = list(args)
        if matched:
            computation_args = computation_args[:-3] + ['hardtanh', [min_value, max_value], '']
            return L[computation_op](*computation_args)
        else:
            out = L[computation_op](*computation_args)
            if is_bf16:
                out = L[prims.convert_element_type.default](out, dtype=torch.float)
            out = L[aten.clamp_max](L[aten.clamp_min](out, min_value), max_value)
            if is_bf16:
                out = L[prims.convert_element_type.default](out, dtype=torch.bfloat16)
            return out
    return fn