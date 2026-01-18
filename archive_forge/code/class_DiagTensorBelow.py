import torch
from copy import deepcopy
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import LoggingTensor
class DiagTensorBelow(WrapperTensor):

    @classmethod
    def get_wrapper_properties(cls, diag, requires_grad=False):
        assert diag.ndim == 1
        return (diag, {'size': diag.size() + diag.size(), 'requires_grad': requires_grad})

    def __init__(self, diag, requires_grad=False):
        self.diag = diag
    handled_ops = {}
    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if not all((issubclass(cls, t) for t in types)):
            return NotImplemented
        fn = cls.handled_ops.get(func.__name__, None)
        if fn:
            return fn(*args, **kwargs or {})
        else:

            def unwrap(e):
                return e.diag.diag() if isinstance(e, DiagTensorBelow) else e

            def wrap(e):
                if isinstance(e, torch.Tensor) and e.ndim == 1:
                    return DiagTensorBelow(e)
                if isinstance(e, torch.Tensor) and e.ndim == 2 and (e.count_nonzero() == e.diag().count_nonzero()):
                    return DiagTensorBelow(e.diag())
                return e
            rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {})))
            return rs

    def __repr__(self):
        return super().__repr__(tensor_contents=f'diag={self.diag}')