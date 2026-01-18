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
@functools.lru_cache(None)
def _mkldnn_weight_pack_init():
    if torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available():
        _register_weight_pack_pass()
        _recover_linear()
        _register_quantization_weight_pack_pass()