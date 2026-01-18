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
def _generate_qconv_weight_prepack_patterns(dtype=torch.float32):
    assert dtype in [torch.float32, torch.bfloat16]
    return (_generate_dequant_convolution_node_pattern(dequantize_per_channel_weight_pattern if dtype == torch.float32 else dequantize_per_channel_to_bf16_weight_pattern, dtype), _generate_dequant_convolution_node_pattern(dequantize_per_channel_clone_weight_pattern if dtype == torch.float32 else dequantize_per_channel_to_bf16_clone_weight_pattern, dtype))