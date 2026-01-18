import types
import numpy as np
import cupy as cp
from cupyx.fallback_mode import notification
def _create_magic_methods():
    """
    Set magic methods of cupy.ndarray as methods of fallback.ndarray.
    """

    def make_method(name):

        def method(self, *args, **kwargs):
            CLASS = cp.ndarray if self._supports_cupy else self._numpy_array.__class__
            _method = getattr(CLASS, name)
            args = (self,) + args
            if self._supports_cupy:
                return _call_cupy(_method, args, kwargs)
            return _call_numpy(_method, args, kwargs)
        method.__doc__ = getattr(np.ndarray, name).__doc__
        return method
    for method in ('__eq__', '__ne__', '__lt__', '__gt__', '__le__', '__ge__', '__neg__', '__pos__', '__abs__', '__invert__', '__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__', '__mod__', '__divmod__', '__pow__', '__lshift__', '__rshift__', '__and__', '__or__', '__xor__', '__iadd__', '__isub__', '__imul__', '__itruediv__', '__ifloordiv__', '__imod__', '__ipow__', '__ilshift__', '__irshift__', '__iand__', '__ior__', '__ixor__', '__matmul__', '__radd__', '__rsub__', '__rmul__', '__rtruediv__', '__rfloordiv__', '__rmod__', '__rdivmod__', '__rpow__', '__rlshift__', '__rrshift__', '__rand__', '__ror__', '__rxor__', '__rmatmul__', '__copy__', '__deepcopy__', '__reduce__', '__iter__', '__len__', '__getitem__', '__setitem__', '__bool__', '__int__', '__float__', '__complex__', '__repr__', '__str__'):
        setattr(ndarray, method, make_method(method))