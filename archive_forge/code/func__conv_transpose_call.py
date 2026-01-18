import functools
import operator
from functools import reduce
from typing import Any, Tuple
import torch
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from .. import ir
from ..lowering import lowerings as L
from ..pattern_matcher import (
from ..virtualized import ops
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern
from .quantization import (
def _conv_transpose_call(users=1):
    return CallFunction(mkldnn._convolution_transpose_pointwise.default, *_conv_transpose_args, _users=users)