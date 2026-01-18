import functools
import itertools
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import Arg, CallFunction, KeywordArg
from .freezing_patterns import register_binary_folding_pattern

    In order to fuse add/sub/mul/div with conv, the dimensions of its
    constant tensor must satisfy the following:
    - with resizing, broadcast to w/ weight/bias tensor shape
    - broadcast to the conv output shape
    It needs to have a shape that can resize to weight/bias
    tensor shape because we need to run the op with the conv
    weights/bias without changing their sizes.
    It needs to broadcast to the conv output shape so that we do
    accidentally change the shape of op output by pre-fusing it
    compared to eager.
    The only dimension value shared by weight/bias/conv output
    is they all contain a dim with value = channels-out. In the
    conv output tensor, this is in the second dimension,
    so the pointwise op tensor may have a second dimension of
    value == channels-out, but all the other dimensions have to be 1
    