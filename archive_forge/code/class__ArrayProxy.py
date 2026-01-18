import numpy
from cupy._core._dtype import get_dtype
import cupy
from cupy._core import _fusion_thread_local
from cupy._core import core
from cupy._core._scalar import get_typename
class _ArrayProxy(_VariableProxy):
    """An abstracted array object passed to the target function.

    Attributes:
        dtype(dtype): The dtype of the array.
        imag(_ArrayProxy): The imaginary part of the array (Not implemented)
        real(_ArrayProxy): The real part of the array (Not implemented)
        ndim(int): The number of dimensions of the array.
    """

    def __repr__(self):
        return "_ArrayProxy([...], dtype='{}', ndim={})".format(self.dtype.char, self.ndim)

    def _inplace_op(self, ufunc, other):
        return ufunc(self, other, self)

    def __iadd__(self, other):
        return self._inplace_op(cupy.add, other)

    def __isub__(self, other):
        return self._inplace_op(cupy.subtract, other)

    def __imul__(self, other):
        return self._inplace_op(cupy.multiply, other)

    def __idiv__(self, other):
        return self._inplace_op(cupy.divide, other)

    def __itruediv__(self, other):
        return self._inplace_op(cupy.true_divide, other)

    def __ifloordiv__(self, other):
        return self._inplace_op(cupy.floor_divide, other)

    def __imod__(self, other):
        return self._inplace_op(cupy.remainder, other)

    def __ipow__(self, other):
        return self._inplace_op(cupy.power, other)

    def __ilshift__(self, other):
        return self._inplace_op(cupy.left_shift, other)

    def __irshift__(self, other):
        return self._inplace_op(cupy.right_shift, other)

    def __iand__(self, other):
        return self._inplace_op(cupy.bitwise_and, other)

    def __ior__(self, other):
        return self._inplace_op(cupy.bitwise_or, other)

    def __ixor__(self, other):
        return self._inplace_op(cupy.bitwise_xor, other)

    def __getitem__(self, index):
        return _fusion_thread_local.call_indexing(self, index)

    def __setitem__(self, slices, value):
        if slices is Ellipsis or (isinstance(slices, slice) and slices == slice(None)):
            _fusion_thread_local.call_ufunc(core.elementwise_copy, value, out=self)
        else:
            raise ValueError('The fusion supports `[...]` or `[:]`.')