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
def _is_valid_quantized_conv2d_optimization_pattern(output_dtype):

    def fn(match):
        if output_dtype is not None:
            qconv_node_after_weight_prepack = filter_nodes(match.nodes, torch.ops.onednn.qconv2d_pointwise)[0]
            return _check_node_kwarg_arg_value(qconv_node_after_weight_prepack, 'output_dtype', 13, output_dtype)
        return True
    return fn