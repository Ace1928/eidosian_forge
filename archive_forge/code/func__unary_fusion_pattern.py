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
def _unary_fusion_pattern(unary_fusion, call_fn, users, is_bf16):
    computation_call = _to_float(call_fn(), users=users) if is_bf16 else call_fn(users=users)
    out = unary_fusion(computation_call)
    return _to_bf16(out) if is_bf16 else out