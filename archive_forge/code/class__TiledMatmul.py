import os
from typing import List, Optional
import torch
import torch.multiprocessing.reductions
from torch.utils._pytree import tree_flatten, tree_unflatten
from typing_extensions import Annotated
from .. import _is_triton_available
from .common import Alias, make_pytorch_operator_for_dispatch_key
class _TiledMatmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, ab_tree_spec, *ab_tree_values):
        ctx.ab_tree_spec = ab_tree_spec
        ctx.save_for_backward(*ab_tree_values)
        a, b = tree_unflatten(list(ab_tree_values), ab_tree_spec)
        c = tiled_matmul_fwd(a, b)
        c_tree_values, c_tree_spec = tree_flatten(c)
        ctx.c_tree_spec = c_tree_spec
        return (c_tree_spec,) + tuple(c_tree_values)

    @staticmethod
    def backward(ctx, _none, *grad_c_tree_values):
        a, b = tree_unflatten(list(ctx.saved_tensors), ctx.ab_tree_spec)
        grad_c = tree_unflatten(list(grad_c_tree_values), ctx.c_tree_spec)
        grad_a = tiled_matmul_fwd(grad_c, _transpose(b))
        grad_b = tiled_matmul_fwd(_transpose(a), grad_c)
        grad_ab_tree_values, grad_ab_tree_spec = tree_flatten((grad_a, grad_b))
        return (None,) + tuple(grad_ab_tree_values)