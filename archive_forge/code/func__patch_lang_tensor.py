import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def _patch_lang_tensor(tensor, builder):
    for name, member in inspect.getmembers(tensor):
        if tl.core.is_builtin(member):
            patch_attr(tensor, name, member, builder)
    tensor.__index__ = lambda self: int(self.handle.data)
    tensor.__bool__ = lambda self: True
    tensor.__str__ = lambda self: str(self.handle.data)
    tensor.__getitem__ = lambda self, slices: self.handle.data.__getitem__(slices)