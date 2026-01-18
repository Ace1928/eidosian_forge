import torch
from torch.fx import GraphModule
from ..utils import (
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.fx.subgraph_rewriter import replace_pattern
from torch._higher_order_ops.out_dtype import out_dtype
from typing import Optional, Callable, Tuple, Any
from dataclasses import dataclass
from functools import partial
@dataclass
class _RewriteInfo:
    """Data needed for rewrite, this includes example inputs, pattern and replacement functions
    and post transformation functions for the exported pattern and replacement GraphModule
    """
    example_inputs: Tuple[Any, ...]
    pattern: Callable
    replacement: Callable
    pattern_post_trans: Optional[Callable[[GraphModule], GraphModule]] = None
    replacement_post_trans: Optional[Callable[[GraphModule], GraphModule]] = None