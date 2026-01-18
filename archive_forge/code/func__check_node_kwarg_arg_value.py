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
def _check_node_kwarg_arg_value(check_node, kwarg_name, args_index, expected_value):
    if kwarg_name in check_node.kwargs:
        actual_value = check_node.kwargs[kwarg_name]
        return actual_value == expected_value
    else:
        assert len(check_node.args) >= args_index + 1
        actual_value = check_node.args[args_index]
        return actual_value == expected_value