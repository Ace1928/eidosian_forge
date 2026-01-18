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
@functools.lru_cache(None)
def _register_quantization_weight_pack_pass():
    for dtype in [torch.float32, torch.bfloat16]:
        _register_dequant_promotion_pass(_may_generate_pattern_with_dtype_convert(dequantize_per_tensor_activation_pattern, KeywordArg('autocast_act_dtype'), dtype != torch.float32), pass_number=0, dtype=dtype)
        weight_prepack_patterns = _generate_qconv_weight_prepack_patterns(dtype)
        for weight_prepack_pattern in weight_prepack_patterns:
            _register_qconv_weight_prepack_pass(weight_prepack_pattern, pass_number=1, dtype=dtype)
        weight_prepack_patterns = _generate_qlinear_weight_prepack_patterns(dtype)
        for weight_prepack_pattern in weight_prepack_patterns:
            _register_qlinear_weight_prepack_pass(weight_prepack_pattern, pass_number=1, dtype=dtype)