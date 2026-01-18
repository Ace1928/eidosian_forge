import copy
import functools
import math
import operator
from typing import Any, Tuple
import torch
from torch._dynamo.utils import counters
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from ..lowering import lowerings as L, require_channels_last
from ..pattern_matcher import Arg, CallFunction, filter_nodes, KeywordArg, ListOf, Match
from ..utils import pad_listlike
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern
def _is_valid_dequant_conv2d_pattern(dtype):

    def _inner(match):
        conv_node = match.output_node()
        assert conv_node.target is aten.convolution.default
        input_meta_value = conv_node.args[0].meta.get('val')
        weight_meta_value = conv_node.args[1].meta.get('val')
        for meta_value in [input_meta_value, weight_meta_value]:
            if meta_value is None or meta_value.device.type != 'cpu' or meta_value.dim() != 4:
                return False
        assert dtype in [torch.float32, torch.bfloat16]
        if dtype == torch.float32:
            mul_node = conv_node.args[0]
        else:
            convert_to_bf16 = conv_node.args[0]
            mul_node = convert_to_bf16.args[0]
        sub_node = mul_node.args[0]
        to_fp32_node = sub_node.args[0]
        assert to_fp32_node.target is prims.convert_element_type.default
        assert sub_node.target is aten.sub.Tensor
        assert mul_node.target is aten.mul.Tensor
        if len(list(to_fp32_node.users)) != 1 or len(list(sub_node.users)) != 1 or len(list(mul_node.users)) != 1:
            return False
        return True
    return _inner