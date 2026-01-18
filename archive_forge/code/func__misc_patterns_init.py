import functools
from typing import Dict, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._ops import OpOverload, OpOverloadPacket
from ..pattern_matcher import fwd_only, register_replacement
@functools.lru_cache(None)
def _misc_patterns_init():
    from .joint_graph import patterns as joint_graph_patterns
    from .post_grad import pass_patterns as post_grad_patterns_all
    post_grad_patterns = post_grad_patterns_all[1]
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    def randperm_index_add_pattern(x, y):
        index = torch.randperm(x.shape[0], device=x.device)[:y.shape[0]]
        return (torch.index_add(x, dim=0, source=y, index=index), index)

    def randperm_index_add_replacement(x, y):
        index = torch.randperm(x.shape[0], device=x.device)[:y.shape[0]]
        return (torch.ops.aten._unsafe_index_put(x, (index,), aten._unsafe_index(x, (index,)) + y, accumulate=False), index)
    register_replacement(randperm_index_add_pattern, randperm_index_add_replacement, [torch.empty(4, 8, device=device), torch.empty(2, 8, device=device)], fwd_only, [post_grad_patterns, joint_graph_patterns])

    def randperm_index_pattern(x, slice_shape):
        index = torch.randperm(x.shape[0], device=x.device)[:slice_shape]
        return (torch.ops.aten.index(x, (index,)), index)

    def randperm_index_replacement(x, slice_shape):
        index = torch.randperm(x.shape[0], device=x.device)[:slice_shape]
        return (torch.ops.aten._unsafe_index(x, (index,)), index)
    pattern = register_replacement(randperm_index_pattern, randperm_index_replacement, [torch.empty(4, 8, device=device)], fwd_only, [post_grad_patterns, joint_graph_patterns], scalar_workaround={'slice_shape': 42})