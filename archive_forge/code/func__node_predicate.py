import logging
import operator
from typing import Callable, List, Optional, Set, Tuple
from functorch import make_fx
import torch
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.decomposition import select_decomp_table
def _node_predicate(node: torch.fx.Node) -> Tuple[bool, str]:
    should_lower, reason = _is_inductor_compatible(node)
    if not should_lower:
        return (should_lower, reason)
    if not node_predicate(node):
        return (False, 'user predicate')
    return (True, '')