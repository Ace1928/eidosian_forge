import functools
from itertools import chain
from typing import List, Optional
import torch
from torch import Tensor
from torch._inductor import utils
from torch.utils._mode_utils import no_dispatch
from torch.utils._triton import has_triton
from ..pattern_matcher import fwd_only, joint_fwd_bwd, Match, register_replacement
@functools.lru_cache(None)
def _pad_mm_init():
    from .joint_graph import patterns
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    dim2a = functools.partial(torch.empty, (4, 4), device=device, requires_grad=True)
    dim2b = functools.partial(torch.empty, (4, 4), device=device, requires_grad=True)
    dim3a = functools.partial(torch.empty, (4, 4, 4), device=device, requires_grad=True)
    dim3b = functools.partial(torch.empty, (4, 4, 4), device=device, requires_grad=True)
    dim1a = functools.partial(torch.empty, 4, device=device, requires_grad=True)
    rep = {'beta': 0.213377, 'alpha': 0.113377}
    for pattern, replacement, args, workaround, extra_check in [(mm_pattern, mm_replace, [dim2a(), dim2b()], {}, should_pad_mm), (bmm_pattern, bmm_replace, [dim3a(), dim3b()], {}, should_pad_bmm), (addmm_pattern, addmm_replace, [dim1a(), dim2a(), dim2b()], rep, should_pad_addmm)]:
        assert isinstance(workaround, dict)
        register_replacement(pattern, replacement, args, joint_fwd_bwd, patterns, extra_check=extra_check, scalar_workaround=workaround)
        register_replacement(pattern, replacement, args, fwd_only, patterns, extra_check=extra_check, scalar_workaround=workaround)