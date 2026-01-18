import functools
import itertools
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import Arg, CallFunction, KeywordArg
from .freezing_patterns import register_binary_folding_pattern
def _op_not_broadcasting_with_conv(weight_tensor, other_tensor):
    weight_shape = weight_tensor.shape
    other_shape = other_tensor.shape
    if len(weight_shape) < len(other_shape):
        return False
    if len(weight_shape) == len(other_shape) + 1:
        for i in reversed(range(len(other_shape))):
            if i == 0 and weight_shape[0] == other_shape[i]:
                continue
            if other_shape[i] != 1:
                return False
    else:
        for i in reversed(range(len(other_shape))):
            if i == 1 and weight_shape[0] == other_shape[i]:
                continue
            if other_shape[i] != 1:
                return False
    return True