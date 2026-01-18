import functools
import itertools
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import Arg, CallFunction, KeywordArg
from .freezing_patterns import register_binary_folding_pattern
def _check_conv_and_broadcast_op(conv_node, other):
    if conv_node.args[1].op != 'get_attr':
        return False
    if conv_node.args[1] is not None and conv_node.args[1].op != 'get_attr':
        return False
    if not isinstance(other, int) and (not isinstance(other, float)) and (other.op != 'get_attr'):
        return False
    if not len(conv_node.args[1].users) == 1:
        return False
    weight_meta_value = conv_node.args[1].meta.get('val')
    if weight_meta_value is None:
        return False
    if not weight_meta_value.is_floating_point():
        return False
    if isinstance(other, torch.fx.Node) and other.op == 'get_attr':
        other_meta_value = other.meta.get('val')
        if not other_meta_value.is_floating_point():
            return False
        if torch.promote_types(other_meta_value.dtype, weight_meta_value.dtype) != weight_meta_value.dtype:
            if not conv_node.meta.get('_allow_conv_mixed_dtype_folding', False):
                return False
            if other_meta_value.dtype != torch.float and weight_meta_value.dtype not in (torch.float16, torch.bfloat16):
                return False
        if not _op_not_broadcasting_with_conv(weight_meta_value, other_meta_value):
            return False
    else:
        return False
    return True