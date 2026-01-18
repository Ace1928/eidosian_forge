import functools
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type
import torch
from ..common import _has_triton21, register_operator
from .attn_bias import (
from .common import AttentionFwOpBase, Context, Inputs, check_lastdim_alignment_stride1
def _get_splitk_kernel(num_groups):
    """
        Kernel _fwd_kernel_splitK needs to be post-processed by unroll_varargs
        to specialize it for a given number of quantization groups N_GROUPS
        before we can apply triton.heuristics and triton.autotune, so we
        don't do them as decorators.
        """
    _fwd_kernel_splitK_unrolled = unroll_varargs(_fwd_kernel_splitK, N=num_groups)
    kernel = triton.heuristics({'BOUNDS_CHECKS_N': lambda args: args['BLOCK_N_PER_SPLIT'] % args['BLOCK_N'] > 0 or args['USE_SEQ_LEN']})(_fwd_kernel_splitK_unrolled)
    return kernel