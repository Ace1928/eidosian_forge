import ast
import re
import sys
import warnings
from ..exceptions import DTypePromotionError
from .multiarray import dtype, array, ndarray, promote_types
class _ctypes:

    def __init__(self, array, ptr=None):
        self._arr = array
        if ctypes:
            self._ctypes = ctypes
            self._data = self._ctypes.c_void_p(ptr)
        else:
            self._ctypes = _missing_ctypes()
            self._data = self._ctypes.c_void_p(ptr)
            self._data._objects = array
        if self._arr.ndim == 0:
            self._zerod = True
        else:
            self._zerod = False

    def data_as(self, obj):
        """
        Return the data pointer cast to a particular c-types object.
        For example, calling ``self._as_parameter_`` is equivalent to
        ``self.data_as(ctypes.c_void_p)``. Perhaps you want to use the data as a
        pointer to a ctypes array of floating-point data:
        ``self.data_as(ctypes.POINTER(ctypes.c_double))``.

        The returned pointer will keep a reference to the array.
        """
        ptr = self._ctypes.cast(self._data, obj)
        ptr._arr = self._arr
        return ptr

    def shape_as(self, obj):
        """
        Return the shape tuple as an array of some other c-types
        type. For example: ``self.shape_as(ctypes.c_short)``.
        """
        if self._zerod:
            return None
        return (obj * self._arr.ndim)(*self._arr.shape)

    def strides_as(self, obj):
        """
        Return the strides tuple as an array of some other
        c-types type. For example: ``self.strides_as(ctypes.c_longlong)``.
        """
        if self._zerod:
            return None
        return (obj * self._arr.ndim)(*self._arr.strides)

    @property
    def data(self):
        """
        A pointer to the memory area of the array as a Python integer.
        This memory area may contain data that is not aligned, or not in correct
        byte-order. The memory area may not even be writeable. The array
        flags and data-type of this array should be respected when passing this
        attribute to arbitrary C-code to avoid trouble that can include Python
        crashing. User Beware! The value of this attribute is exactly the same
        as ``self._array_interface_['data'][0]``.

        Note that unlike ``data_as``, a reference will not be kept to the array:
        code like ``ctypes.c_void_p((a + b).ctypes.data)`` will result in a
        pointer to a deallocated array, and should be spelt
        ``(a + b).ctypes.data_as(ctypes.c_void_p)``
        """
        return self._data.value

    @property
    def shape(self):
        """
        (c_intp*self.ndim): A ctypes array of length self.ndim where
        the basetype is the C-integer corresponding to ``dtype('p')`` on this
        platform (see `~numpy.ctypeslib.c_intp`). This base-type could be
        `ctypes.c_int`, `ctypes.c_long`, or `ctypes.c_longlong` depending on
        the platform. The ctypes array contains the shape of
        the underlying array.
        """
        return self.shape_as(_getintp_ctype())

    @property
    def strides(self):
        """
        (c_intp*self.ndim): A ctypes array of length self.ndim where
        the basetype is the same as for the shape attribute. This ctypes array
        contains the strides information from the underlying array. This strides
        information is important for showing how many bytes must be jumped to
        get to the next element in the array.
        """
        return self.strides_as(_getintp_ctype())

    @property
    def _as_parameter_(self):
        """
        Overrides the ctypes semi-magic method

        Enables `c_func(some_array.ctypes)`
        """
        return self.data_as(ctypes.c_void_p)

    def get_data(self):
        """Deprecated getter for the `_ctypes.data` property.

        .. deprecated:: 1.21
        """
        warnings.warn('"get_data" is deprecated. Use "data" instead', DeprecationWarning, stacklevel=2)
        return self.data

    def get_shape(self):
        """Deprecated getter for the `_ctypes.shape` property.

        .. deprecated:: 1.21
        """
        warnings.warn('"get_shape" is deprecated. Use "shape" instead', DeprecationWarning, stacklevel=2)
        return self.shape

    def get_strides(self):
        """Deprecated getter for the `_ctypes.strides` property.

        .. deprecated:: 1.21
        """
        warnings.warn('"get_strides" is deprecated. Use "strides" instead', DeprecationWarning, stacklevel=2)
        return self.strides

    def get_as_parameter(self):
        """Deprecated getter for the `_ctypes._as_parameter_` property.

        .. deprecated:: 1.21
        """
        warnings.warn('"get_as_parameter" is deprecated. Use "_as_parameter_" instead', DeprecationWarning, stacklevel=2)
        return self._as_parameter_