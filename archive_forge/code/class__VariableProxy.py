import numpy
from cupy._core._dtype import get_dtype
import cupy
from cupy._core import _fusion_thread_local
from cupy._core import core
from cupy._core._scalar import get_typename
class _VariableProxy:
    """Abstracted array/scalar object passed to the target function.
    """

    def __init__(self, content):
        assert isinstance(content, cupy._core._fusion_variable._TraceVariable)
        self.content = content

    def __neg__(self):
        return cupy.negative(self)

    def __add__(self, other):
        return cupy.add(self, other)

    def __radd__(self, other):
        return cupy.add(other, self)

    def __sub__(self, other):
        return cupy.subtract(self, other)

    def __rsub__(self, other):
        return cupy.subtract(other, self)

    def __mul__(self, other):
        return cupy.multiply(self, other)

    def __rmul__(self, other):
        return cupy.multiply(other, self)

    def __div__(self, other):
        return cupy.divide(self, other)

    def __rdiv__(self, other):
        return cupy.divide(other, self)

    def __truediv__(self, other):
        return cupy.true_divide(self, other)

    def __rtruediv__(self, other):
        return cupy.true_divide(other, self)

    def __floordiv__(self, other):
        return cupy.floor_divide(self, other)

    def __rfloordiv__(self, other):
        return cupy.floor_divide(other, self)

    def __mod__(self, other):
        return cupy.remainder(self, other)

    def __rmod__(self, other):
        return cupy.remainder(other, self)

    def __pow__(self, other):
        return cupy.power(self, other)

    def __lshift__(self, other):
        return cupy.left_shift(self, other)

    def __rlshift__(self, other):
        return cupy.left_shift(other, self)

    def __rshift__(self, other):
        return cupy.right_shift(self, other)

    def __rrshift__(self, other):
        return cupy.right_shift(other, self)

    def __invert__(self):
        return cupy.invert(self)

    def __and__(self, other):
        return cupy.bitwise_and(self, other)

    def __rand__(self, other):
        return cupy.bitwise_and(other, self)

    def __or__(self, other):
        return cupy.bitwise_or(self, other)

    def __ror__(self, other):
        return cupy.bitwise_or(other, self)

    def __xor__(self, other):
        return cupy.bitwise_xor(self, other)

    def __rxor__(self, other):
        return cupy.bitwise_xor(other, self)

    def __lt__(self, other):
        return cupy.less(self, other)

    def __le__(self, other):
        return cupy.less_equal(self, other)

    def __eq__(self, other):
        return cupy.equal(self, other)

    def __ne__(self, other):
        return cupy.not_equal(self, other)

    def __ge__(self, other):
        return cupy.greater_equal(self, other)

    def __gt__(self, other):
        return cupy.greater(self, other)

    def copy(self):
        return cupy.copy(self)

    def astype(self, dtype, order=None, casting=None, subok=None, copy=True):
        dtype = get_dtype(dtype)
        if order is not None:
            raise TypeError('order is not supported yet')
        if casting is not None:
            raise TypeError('casting is not supported yet')
        if subok is not None:
            raise TypeError('subok is not supported yet')
        if not copy and self.dtype == dtype:
            return self
        if _dtype_to_astype_dict is None:
            _set_dtype_to_astype_dict()
        return _dtype_to_astype_dict[dtype](self)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        return cupy.sum(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def prod(self, axis=None, dtype=None, out=None, keepdims=False):
        return cupy.prod(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def max(self, axis=None, out=None, keepdims=False):
        return cupy.max(self, axis=axis, out=out, keepdims=keepdims)

    def min(self, axis=None, out=None, keepdims=False):
        return cupy.min(self, axis=axis, out=out, keepdims=keepdims)

    def all(self, axis=None, out=None, keepdims=False):
        return cupy.all(self, axis=axis, out=out, keepdims=keepdims)

    def any(self, axis=None, out=None, keepdims=False):
        return cupy.any(self, axis=axis, out=out, keepdims=keepdims)

    @property
    def dtype(self):
        return self.content.dtype

    @property
    def ndim(self):
        return self.content.ndim

    @property
    def shape(self):
        raise NotImplementedError('`shape` is not supported, currently.')