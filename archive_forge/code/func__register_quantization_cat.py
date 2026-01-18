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
def _register_quantization_cat():
    dequantize_cat_pattern = CallFunction(aten.cat.default, ListOf(_raw_dequantize_per_tensor_activation_pattern), KeywordArg('dim'))
    _register_quantized_cat_lowering(generate_pattern_with_output_quant(dequantize_cat_pattern), aten.cat)