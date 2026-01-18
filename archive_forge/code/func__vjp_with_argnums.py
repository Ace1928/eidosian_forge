from typing import Callable, Union, Tuple, List, Any, Optional
import torch
from functools import partial, wraps
import contextlib
from torch.utils._pytree import (
from torch.utils import _pytree as pytree
from torch.fx.experimental import const_fold
from torch.fx.experimental.proxy_tensor import make_fx
import torch.autograd.forward_ad as fwAD
from torch._subclasses.functional_tensor import FunctionalTensor
from .vmap import doesnt_support_saved_tensors_hooks, get_chunk_sizes
from .apis import vmap
from torch._C._functorch import (
from torch._functorch.utils import exposed_in, argnums_t
@doesnt_support_saved_tensors_hooks
def _vjp_with_argnums(func: Callable, *primals, argnums: Optional[argnums_t]=None, has_aux: bool=False):
    level = _grad_increment_nesting()
    try:
        with torch.enable_grad():
            primals = _wrap_all_tensors(primals, level)
            if argnums is None:
                diff_primals = _create_differentiable(primals, level)
            else:
                diff_primals = _slice_argnums(primals, argnums, as_tuple=False)
                tree_map_(partial(_create_differentiable, level=level), diff_primals)
            primals_out = func(*primals)
            if has_aux:
                if not (isinstance(primals_out, tuple) and len(primals_out) == 2):
                    raise RuntimeError('vjp(f, *primals): output of function f should be a tuple: (output, aux) if has_aux is True')
                primals_out, aux = primals_out
                aux = _undo_create_differentiable(aux, level)
            flat_primals_out, primals_out_spec = tree_flatten(primals_out)
            assert_non_empty_tensor_output(flat_primals_out, 'vjp(f, *primals)')
            flat_diff_primals, primals_spec = tree_flatten(diff_primals)
            results = _undo_create_differentiable(primals_out, level)
            for primal_out in flat_primals_out:
                assert isinstance(primal_out, torch.Tensor)
                if primal_out.is_floating_point() or primal_out.is_complex():
                    continue
                raise RuntimeError(f'vjp(f, ...): All outputs of f must be floating-point or complex Tensors, got Tensor with dtype {primal_out.dtype}')

        def wrapper(cotangents, retain_graph=True, create_graph=None):
            if create_graph is None:
                create_graph = torch.is_grad_enabled()
            flat_cotangents, cotangents_spec = tree_flatten(cotangents)
            if primals_out_spec != cotangents_spec:
                raise RuntimeError(f'Expected pytree structure of cotangents to be the same as pytree structure of outputs to the function. cotangents: {treespec_pprint(cotangents_spec)}, primal output: {treespec_pprint(primals_out_spec)}')
            result = _autograd_grad(flat_primals_out, flat_diff_primals, flat_cotangents, retain_graph=retain_graph, create_graph=create_graph)
            return tree_unflatten(result, primals_spec)
    finally:
        _grad_decrement_nesting()
    if has_aux:
        return (results, wrapper, aux)
    else:
        return (results, wrapper)