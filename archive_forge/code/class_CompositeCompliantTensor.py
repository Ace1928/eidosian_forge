import torch
from torch import Tensor
import itertools
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from torch.utils import _pytree as pytree
from functools import partial
from torch.utils._mode_utils import no_dispatch, all_same_mode
import torch.autograd.forward_ad as fwAD
from typing import Callable
import re
class CompositeCompliantTensor(torch.Tensor):
    elem: torch.Tensor
    __slots__ = ['elem']
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, elem, mode, *args, **kwargs):
        assert type(elem) is not cls, 'Wrapping a CompositeCompliantTensor in a CompositeCompliantTensor is not supported'
        r = torch.Tensor._make_wrapper_subclass(cls, elem.size(), dtype=elem.dtype, layout=elem.layout, device=elem.device, requires_grad=elem.requires_grad, strides=elem.stride(), storage_offset=elem.storage_offset())
        if elem.requires_grad:
            tmp = torch.empty_strided(elem.shape, elem.stride(), dtype=elem.dtype, device=elem.device, layout=elem.layout, requires_grad=False)
            tmp.copy_(elem.detach())
            r.elem = tmp
        else:
            r.elem = elem
        assert r.stride() == r.elem.stride()
        torch._C._set_conj(r, r.elem.is_conj())
        torch._C._set_neg(r, r.elem.is_neg())
        r.mode = mode
        return r

    def __repr__(self):
        return f'CompositeCompliantTensor({self.elem})'

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        all_args = pytree.arg_tree_leaves(*args, **kwargs or {})
        modes = tuple((e.mode for e in all_args if isinstance(e, CompositeCompliantTensor)))
        if not all_same_mode(modes):
            raise RuntimeError('Multiple CompositeCompliantTensorModes NYI')
        with modes[0]:
            return func(*args, **kwargs)