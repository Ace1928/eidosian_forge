import torch
import torch.fx
import warnings
import functools
import builtins
from typing import Any, Callable, Dict, Optional, Union
class MetaProxy(torch.fx.Proxy):

    def install_tensor_meta(self, tensor_meta):
        self._tensor_meta = tensor_meta

    def size(self, dim=None):
        if hasattr(self, '_tensor_meta') and self._tensor_meta is not None:
            return self._tensor_meta.size(*([dim] if dim else []))
        return self.tracer.create_proxy('call_method', 'size', (self, dim) if dim else (self,), {})

    def dim(self):
        if hasattr(self, '_tensor_meta') and self._tensor_meta is not None:
            return self._tensor_meta.dim()
        return self.tracer.create_proxy('call_method', 'dim', (self,), {})

    @property
    def shape(self):
        if hasattr(self, '_tensor_meta') and self._tensor_meta is not None:
            return self._tensor_meta.shape
        return self.tracer.create_proxy('call_function', builtins.getattr, (self, 'shape'), {})

    @property
    def dtype(self):
        if hasattr(self, '_tensor_meta') and self._tensor_meta is not None:
            return self._tensor_meta.dtype
        return self.tracer.create_proxy('call_function', builtins.getattr, (self, 'dtype'), {})

    @property
    def device(self):
        return MetaDeviceAttribute(self, 'device')

    def __getattr__(self, k):
        if k == '_tensor_meta':
            return self.__getattribute__(k)
        return MetaAttribute(self, k)