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
def _is_valid_quantized_conv_binary_optimization_pattern(output_dtype):

    def fn(match):
        qconv2d_node_after_weight_prepack = filter_nodes(match.nodes, torch.ops.onednn.qconv2d_pointwise)[0]
        if len(qconv2d_node_after_weight_prepack.users) != 1:
            return False
        if output_dtype is not None:
            binary_node_inputs = list(qconv2d_node_after_weight_prepack.users)[0].args
            assert len(binary_node_inputs) == 2, 'Expects binary node with 2 inputs'
            extra_input_node = None
            for arg in binary_node_inputs:
                if arg != qconv2d_node_after_weight_prepack:
                    extra_input_node = arg
                    break
            assert extra_input_node is not None
            if not isinstance(extra_input_node, torch.fx.Node) or extra_input_node.target != aten.mul.Tensor:
                return False
        return True
    return fn