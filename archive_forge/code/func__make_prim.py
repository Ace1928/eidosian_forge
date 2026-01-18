import contextlib
import itertools
import operator
import weakref
from enum import Enum
from functools import partial, reduce
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union
import torch
import torch._prims_common as utils
import torch.library
from torch import sym_float, Tensor, TypedStorage
from torch._C import _get_default_device
from torch._prims.debug_prims import register_debug_prims
from torch._prims.rng_prims import register_rng_prims
from torch._prims_common import (
from torch._prims_common.wrappers import backwards_not_supported
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.overrides import handle_torch_function, has_torch_function
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten
def _make_prim(*, schema: str, return_type: Union[RETURN_TYPE, Tuple[RETURN_TYPE, ...]], meta: Callable, impl_aten: Callable, doc: str, tags: Optional[Sequence[torch.Tag]]=None):
    """
    Creates a primitive operation.

    """
    prim.define(schema, tags=torch.Tag.pt2_compliant_tag)

    def _prim_impl(*args, **kwargs):
        meta(*args, **kwargs)
        return impl_aten(*args, **kwargs)

    def _autograd_impl(*args, **kwargs):
        return backwards_not_supported(_prim)(*args, **kwargs)

    def _backend_select_impl(*args, **kwargs):
        if kwargs.get('device') and kwargs['device'].type == 'meta':
            return meta(*args, **kwargs)
        if any((isinstance(x, torch.device) and x.type == 'meta' for x in args)):
            return meta(*args, **kwargs)
        else:
            return _prim_impl(*args, **kwargs)
    name = schema.split('(')[0]
    prim_impl.impl(name, _prim_impl)
    prim_autograd_impl.impl(name, _autograd_impl)
    prim_meta_impl.impl(name, meta)
    _prim_packet = getattr(torch._ops.ops.prims, name)
    _prim = _prim_packet.default
    if tags:
        _prim._tags = tags
    from torch._subclasses.fake_tensor import contains_tensor_types
    if not any((contains_tensor_types(a.type) for a in _prim._schema.arguments)) or str(_prim) in ['prims.device_put.default']:
        prim_backend_select_impl.impl(name, _backend_select_impl)
    for p in (_prim_packet, _prim):
        p.__doc__ = doc
        p.return_type = return_type
        p.schema = schema
        p.prim_impl = _prim_impl
        p.prim_meta_impl = meta
        p.impl_aten = impl_aten
    return _prim