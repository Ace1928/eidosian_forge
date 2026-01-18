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
def _generate_dequant_linear_node_pattern(_dequant_per_channel_pattern, dtype=torch.float32):
    t_pattern = CallFunction(aten.permute.default, _may_generate_pattern_with_dtype_convert(_dequant_per_channel_pattern, KeywordArg('autocast_wgt_dtype'), dtype != torch.float32), KeywordArg('permute_axes'))
    dequant_linear_bias_pattern = CallFunction(aten.addmm.default, KeywordArg('b'), _may_generate_pattern_with_dtype_convert(dequantize_per_tensor_activation_pattern, KeywordArg('autocast_act_dtype'), dtype != torch.float32), t_pattern)
    dequant_linear_no_bias_pattern = CallFunction(aten.mm.default, _may_generate_pattern_with_dtype_convert(dequantize_per_tensor_activation_pattern, KeywordArg('autocast_act_dtype'), dtype != torch.float32), t_pattern)
    return (dequant_linear_bias_pattern, dequant_linear_no_bias_pattern)