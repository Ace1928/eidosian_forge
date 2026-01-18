import logging
import operator
from typing import Callable, List, Optional, Set, Tuple
from functorch import make_fx
import torch
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.decomposition import select_decomp_table
def _is_inductor_compatible(node: torch.fx.Node) -> Tuple[bool, str]:
    if node.target in (torch.ops.aten._fused_adam_.default, torch.ops.aten._fused_adam.default, torch.ops.aten._foreach_add_.Scalar, torch.ops.aten._foreach_add.Scalar):
        return (False, 'fused adam is not supported yet')
    if node.target == torch.ops.aten.flatten.using_ints:
        return (True, '')
    if isinstance(node.target, torch._ops.OpOverload):
        if not node.target.has_kernel_for_dispatch_key(torch._C.DispatchKey.Meta):
            return (False, f"{node.target} doesn't have a meta kernel registered")
    return (True, '')