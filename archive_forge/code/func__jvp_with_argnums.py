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
def _jvp_with_argnums(func: Callable, primals: Any, tangents: Any, argnums: Optional[argnums_t], *, strict: bool=False, has_aux: bool):
    if not isinstance(primals, tuple):
        raise RuntimeError(f'{jvp_str}: Expected primals to be a tuple. E.g. it should be valid to call f(*primals).')
    diff_args = primals if argnums is None else _slice_argnums(primals, argnums)
    flat_primals, primals_spec = tree_flatten(diff_args)
    flat_tangents, tangents_spec = tree_flatten(tangents)
    if primals_spec != tangents_spec:
        raise RuntimeError(f'{jvp_str}: Expected primals and tangents to have the same python structure. For example, if primals is a tuple of 3 tensors, tangents also must be. Got primals with structure {primals_spec} and tangents with structure {tangents_spec}')
    assert_non_empty_list_of_tensors(flat_primals, jvp_str, 'primals')
    assert_non_empty_list_of_tensors(flat_tangents, jvp_str, 'tangents')
    level = _jvp_increment_nesting()
    try:
        global JVP_NESTING
        JVP_NESTING += 1
        with fwAD._set_fwd_grad_enabled(True):
            ctx = fwAD.dual_level if JVP_NESTING == 1 else noop
            with ctx():
                flat_duals = tuple((fwAD.make_dual(p, t) for p, t in zip(flat_primals, flat_tangents)))
                duals = tree_unflatten(flat_duals, primals_spec)
                if argnums is not None:
                    primals = _wrap_all_tensors(primals, level)
                    duals = _replace_args(primals, duals, argnums)
                result_duals = func(*duals)
                if has_aux:
                    if not (isinstance(result_duals, tuple) and len(result_duals) == 2):
                        raise RuntimeError(f'{jvp_str}: output of function f should be a tuple: (output, aux) if has_aux is True')
                    result_duals, aux = result_duals
                    aux = _undo_create_differentiable(aux, level)
                result_duals, spec = tree_flatten(result_duals)
                assert_non_empty_tensor_output(result_duals, jvp_str)
                primals_out, tangents_out = zip(*[safe_unpack_dual(dual, strict) for dual in result_duals])
                primals_out = tree_map(partial(_undo_create_differentiable, level=level), primals_out)
                tangents_out = tree_map(partial(_undo_create_differentiable, level=level), tangents_out)
                primals_out_unflatten = tree_unflatten(primals_out, spec)
                tangents_out_unflatten = tree_unflatten(tangents_out, spec)
                if has_aux:
                    return (primals_out_unflatten, tangents_out_unflatten, aux)
                return (primals_out_unflatten, tangents_out_unflatten)
    finally:
        _jvp_decrement_nesting()
        JVP_NESTING -= 1