import torch
from copy import deepcopy
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import LoggingTensor
class WrapperTensor(torch.Tensor):

    @staticmethod
    def __new__(cls, *args, **kwargs):
        t, kwargs = cls.get_wrapper_properties(*args, **kwargs)
        if 'size' not in kwargs:
            size = t.size()
        else:
            size = kwargs['size']
            del kwargs['size']
        if 'dtype' not in kwargs:
            kwargs['dtype'] = t.dtype
        if 'layout' not in kwargs:
            kwargs['layout'] = t.layout
        if 'device' not in kwargs:
            kwargs['device'] = t.device
        if 'requires_grad' not in kwargs:
            kwargs['requires_grad'] = False
        wrapper = torch.Tensor._make_wrapper_subclass(cls, size, **kwargs)
        wrapper._validate_methods()
        return wrapper

    @classmethod
    def get_wrapper_properties(cls, *args, **kwargs):
        raise NotImplementedError('You need to implement get_wrapper_properties')

    def _validate_methods(self):
        forbidden_overrides = ['size', 'stride', 'dtype', 'layout', 'device', 'requires_grad']
        for el in forbidden_overrides:
            if getattr(self.__class__, el) is not getattr(torch.Tensor, el):
                raise RuntimeError(f'Subclass {self.__class__.__name__} is overwriting the property {el} but this is not allowed as such change would not be reflected to c++ callers.')