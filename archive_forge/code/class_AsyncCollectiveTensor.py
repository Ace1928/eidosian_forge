import warnings
import sys
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import Tuple, Union, List, Optional, cast, TYPE_CHECKING
from . import _functional_collectives_impl as fun_col_impl
from ._functional_collectives_impl import _register_tensor_wrapper
from torch.fx.experimental.proxy_tensor import (
from torch._custom_ops import impl_abstract
from torch.distributed.distributed_c10d import (
class AsyncCollectiveTensor(torch.Tensor):
    """
    A Tensor wrapper subclass that is used to trigger a call to wait
    prior to first use of the underlying tensor.
    Use it inside functional collective pytorch wrappers like the following:
    def functional_collective(self, group, tag):
        tag, rankset, group_size = _expand_group(group, tag)
        tensor = torch.ops.c10d_functional.{collective}(self, tag, rankset, group_size)
        return _maybe_wrap_tensor(tensor)
    """
    elem: torch.Tensor
    __slots__ = ['elem']
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, elem: torch.Tensor):
        r = torch.Tensor._make_wrapper_subclass(cls, elem.size(), strides=elem.stride(), storage_offset=elem.storage_offset(), dtype=elem.dtype, layout=elem.layout, device=elem.device, requires_grad=False)
        r.elem = elem
        return r

    def __tensor_flatten__(self):
        return (['elem'], None)

    def tolist(self):
        wait_tensor(self.elem)
        return self.elem.tolist()

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta):
        assert meta is None
        elem = inner_tensors['elem']
        return AsyncCollectiveTensor(elem)

    def __repr__(self):
        wait_tensor(self.elem)
        return f'AsyncCollectiveTensor({self.elem})'

    def trigger_wait(self):
        wait_tensor(self.elem)
        return self

    def wait(self) -> torch.Tensor:
        wait_tensor(self.elem)
        return self.elem

    def _get_acs_underlying_tensor(self):
        """This method enables  _functional_collectives_impl to test if a tensor is an ACS"""
        return self.elem

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        is_view_op = _is_view_op(func)

        def unwrap(e: AsyncCollectiveTensor):
            if not is_view_op:
                wait_tensor(e.elem)
            return e.elem

        def wrap(e: torch.Tensor):
            assert not isinstance(e, AsyncCollectiveTensor)
            res = AsyncCollectiveTensor(e)
            _register_tensor_wrapper(res)
            return res
        unwrapped_args = tree_map_only(AsyncCollectiveTensor, unwrap, args)
        unwrapped_kwargs = tree_map_only(AsyncCollectiveTensor, unwrap, kwargs)
        out = func(*unwrapped_args, **unwrapped_kwargs)
        if is_view_op:
            out = tree_map_only(torch.Tensor, wrap, out)
        return out

    def numpy(self):
        return self.wait().numpy()