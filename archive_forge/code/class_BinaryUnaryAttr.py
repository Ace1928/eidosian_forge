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
class BinaryUnaryAttr:

    def __init__(self, binary_op_name: str, alpha=None, unary_op_name: str='none', scalars_attr=None, algorithm_attr=None):
        self.binary_op_name = binary_op_name
        self.alpha = alpha if alpha else 1.0
        self.unary_op_name = unary_op_name
        self.scalars_attr = scalars_attr if scalars_attr else []
        self.algorithm_attr = algorithm_attr if algorithm_attr else ''