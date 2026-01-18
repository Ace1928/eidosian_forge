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
def _generate_dequant_convolution_node_pattern(_dequant_per_channel_pattern, dtype=torch.float32):
    assert dtype in [torch.float32, torch.bfloat16]
    dequant_convolution_node_pattern = CallFunction(aten.convolution.default, _may_generate_pattern_with_dtype_convert(dequantize_per_tensor_activation_pattern, KeywordArg('autocast_act_dtype'), dtype != torch.float32), _dequant_per_channel_pattern, KeywordArg('b'), KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('is_transposed'), KeywordArg('out_padding'), KeywordArg('groups'))
    return dequant_convolution_node_pattern