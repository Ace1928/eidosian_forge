import functools
import itertools
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import Arg, CallFunction, KeywordArg
from .freezing_patterns import register_binary_folding_pattern
def _is_foldable_pattern(match):
    binary_node = match.output_node()
    computation_node = binary_node.args[0]
    other = binary_node.args[1]
    if binary_node.args[0].target not in _computation_ops:
        computation_node = binary_node.args[1]
        other = binary_node.args[0]
    if binary_node.args[0].target == aten.convolution.default:
        return _check_conv_and_broadcast_op(computation_node, other)
    return False