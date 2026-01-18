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
def _may_generate_pattern_with_dtype_convert(pattern, dtype=Arg(), dtype_convert=True):
    if dtype_convert:
        return CallFunction(prims.convert_element_type.default, pattern, dtype)
    else:
        return pattern