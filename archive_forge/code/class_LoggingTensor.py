import torch
from torch.utils._pytree import tree_map
from typing import Iterator, List, Optional
import logging
import contextlib
import itertools
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.weak import WeakTensorKeyDictionary
import functools
from torch._C._profiler import gather_traceback, symbolize_tracebacks
class LoggingTensor(torch.Tensor):
    elem: torch.Tensor
    __slots__ = ['elem']
    context = contextlib.nullcontext
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        r = torch.Tensor._make_wrapper_subclass(cls, elem.size(), strides=elem.stride(), storage_offset=elem.storage_offset(), dtype=elem.dtype, layout=elem.layout, device=elem.device, requires_grad=kwargs.get('requires_grad', False))
        r.elem = elem.detach() if r.requires_grad else elem
        return r

    def __repr__(self):
        return super().__repr__(tensor_contents=f'{self.elem}')

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

        def unwrap(e):
            return e.elem if isinstance(e, cls) else e

        def wrap(e):
            return cls(e) if isinstance(e, torch.Tensor) else e
        with cls.context():
            rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
        logging.getLogger('LoggingTensor').info(f'{func.__module__}.{func.__name__}', args, kwargs, rs)
        return rs