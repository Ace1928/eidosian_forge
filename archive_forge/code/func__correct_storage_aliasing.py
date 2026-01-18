import contextlib
from typing import Optional, Union, List, Set, Dict, Any
import warnings
from dataclasses import dataclass
import torch
import torchgen
from torch._C import _len_torch_dispatch_stack, _get_dispatch_stack_at,\
def _correct_storage_aliasing(func, schema_info, args, outs):
    """
    Given: an OpOverload, a SchemaInfo (cached information from torchgen about schema),
    and the inputs/outputs to the OpOverload,
    this function checks to see if func is a view operator
    (by checking if any of the outputs in the op's schema
     are immutable aliases of inputs).
    If so, this function manually aliases the storage of the output tensor
    with its corresponding input tensor alias.
    It does this by unsafely overwriting the storage field of the output tensor
    to be the same storage as the input.
    """
    assert isinstance(func, torch._ops.OpOverload)
    assert isinstance(args, tuple)
    assert isinstance(outs, (list, tuple))
    flat_outs = torch.utils._pytree.tree_leaves(outs)

    def alias_non_inplace_storage(arg, ret):
        if is_traceable_wrapper_subclass(arg) or is_traceable_wrapper_subclass(ret):
            ret_list = ret if isinstance(ret, list) else [ret]
            for r in ret_list:
                assert type(arg) == type(r), f'Called {str(func)} with input of type {type(arg)}\nand output of type {type(ret)}. But expected types to match.'
        with torch.utils._mode_utils.no_dispatch():
            meta_in_tls = torch._C._meta_in_tls_dispatch_include()
            torch._C._set_meta_in_tls_dispatch_include(True)
            try:
                if isinstance(ret, list):
                    for r in ret:
                        torch.ops.aten.set_.source_Storage_storage_offset(r, arg.untyped_storage(), r.storage_offset(), r.shape)
                else:
                    assert isinstance(ret, torch.Tensor), f'type: {type(ret)}'
                    torch.ops.aten.set_.source_Storage_storage_offset(ret, arg.untyped_storage(), ret.storage_offset(), ret.shape)
            finally:
                torch._C._set_meta_in_tls_dispatch_include(meta_in_tls)

    def is_read_only_alias_match(arg, ret):
        shared_aliases = arg.alias_set & ret.alias_set
        return len(shared_aliases) > 0 and (not arg.is_write)
    num_args = len(func._schema.arguments)
    num_returns = len(func._schema.returns)
    for arg_idx in range(num_args):
        for return_idx in range(num_returns):
            if is_read_only_alias_match(schema_info.args[arg_idx], schema_info.outs[return_idx]):
                alias_non_inplace_storage(args[arg_idx], outs[return_idx])