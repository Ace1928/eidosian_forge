import math
import functools
import operator
import copy
from ctypes import c_void_p
import numpy as np
import numba
from numba import _devicearray
from numba.cuda.cudadrv import devices
from numba.cuda.cudadrv import driver as _driver
from numba.core import types, config
from numba.np.unsafe.ndarray import to_fixed_tuple
from numba.misc import dummyarray
from numba.np import numpy_support
from numba.cuda.api_util import prepare_shape_strides_dtype
from numba.core.errors import NumbaPerformanceWarning
from warnings import warn
class DeviceNDArray(DeviceNDArrayBase):
    """
    An on-GPU array type
    """

    def is_f_contiguous(self):
        """
        Return true if the array is Fortran-contiguous.
        """
        return self._dummy.is_f_contig

    @property
    def flags(self):
        """
        For `numpy.ndarray` compatibility. Ideally this would return a
        `np.core.multiarray.flagsobj`, but that needs to be constructed
        with an existing `numpy.ndarray` (as the C- and F- contiguous flags
        aren't writeable).
        """
        return dict(self._dummy.flags)

    def is_c_contiguous(self):
        """
        Return true if the array is C-contiguous.
        """
        return self._dummy.is_c_contig

    def __array__(self, dtype=None):
        """
        :return: an `numpy.ndarray`, so copies to the host.
        """
        if dtype:
            return self.copy_to_host().__array__(dtype)
        else:
            return self.copy_to_host().__array__()

    def __len__(self):
        return self.shape[0]

    def reshape(self, *newshape, **kws):
        """
        Reshape the array without changing its contents, similarly to
        :meth:`numpy.ndarray.reshape`. Example::

            d_arr = d_arr.reshape(20, 50, order='F')
        """
        if len(newshape) == 1 and isinstance(newshape[0], (tuple, list)):
            newshape = newshape[0]
        cls = type(self)
        if newshape == self.shape:
            return cls(shape=self.shape, strides=self.strides, dtype=self.dtype, gpu_data=self.gpu_data)
        newarr, extents = self._dummy.reshape(*newshape, **kws)
        if extents == [self._dummy.extent]:
            return cls(shape=newarr.shape, strides=newarr.strides, dtype=self.dtype, gpu_data=self.gpu_data)
        else:
            raise NotImplementedError('operation requires copying')

    def ravel(self, order='C', stream=0):
        """
        Flattens a contiguous array without changing its contents, similar to
        :meth:`numpy.ndarray.ravel`. If the array is not contiguous, raises an
        exception.
        """
        stream = self._default_stream(stream)
        cls = type(self)
        newarr, extents = self._dummy.ravel(order=order)
        if extents == [self._dummy.extent]:
            return cls(shape=newarr.shape, strides=newarr.strides, dtype=self.dtype, gpu_data=self.gpu_data, stream=stream)
        else:
            raise NotImplementedError('operation requires copying')

    @devices.require_context
    def __getitem__(self, item):
        return self._do_getitem(item)

    @devices.require_context
    def getitem(self, item, stream=0):
        """Do `__getitem__(item)` with CUDA stream
        """
        return self._do_getitem(item, stream)

    def _do_getitem(self, item, stream=0):
        stream = self._default_stream(stream)
        arr = self._dummy.__getitem__(item)
        extents = list(arr.iter_contiguous_extent())
        cls = type(self)
        if len(extents) == 1:
            newdata = self.gpu_data.view(*extents[0])
            if not arr.is_array:
                if self.dtype.names is not None:
                    return DeviceRecord(dtype=self.dtype, stream=stream, gpu_data=newdata)
                else:
                    hostary = np.empty(1, dtype=self.dtype)
                    _driver.device_to_host(dst=hostary, src=newdata, size=self._dummy.itemsize, stream=stream)
                return hostary[0]
            else:
                return cls(shape=arr.shape, strides=arr.strides, dtype=self.dtype, gpu_data=newdata, stream=stream)
        else:
            newdata = self.gpu_data.view(*arr.extent)
            return cls(shape=arr.shape, strides=arr.strides, dtype=self.dtype, gpu_data=newdata, stream=stream)

    @devices.require_context
    def __setitem__(self, key, value):
        return self._do_setitem(key, value)

    @devices.require_context
    def setitem(self, key, value, stream=0):
        """Do `__setitem__(key, value)` with CUDA stream
        """
        return self._do_setitem(key, value, stream=stream)

    def _do_setitem(self, key, value, stream=0):
        stream = self._default_stream(stream)
        synchronous = not stream
        if synchronous:
            ctx = devices.get_context()
            stream = ctx.get_default_stream()
        arr = self._dummy.__getitem__(key)
        newdata = self.gpu_data.view(*arr.extent)
        if isinstance(arr, dummyarray.Element):
            shape = ()
            strides = ()
        else:
            shape = arr.shape
            strides = arr.strides
        lhs = type(self)(shape=shape, strides=strides, dtype=self.dtype, gpu_data=newdata, stream=stream)
        rhs, _ = auto_device(value, stream=stream, user_explicit=True)
        if rhs.ndim > lhs.ndim:
            raise ValueError("Can't assign %s-D array to %s-D self" % (rhs.ndim, lhs.ndim))
        rhs_shape = np.ones(lhs.ndim, dtype=np.int64)
        rhs_shape[lhs.ndim - rhs.ndim:] = rhs.shape
        rhs = rhs.reshape(*rhs_shape)
        for i, (l, r) in enumerate(zip(lhs.shape, rhs.shape)):
            if r != 1 and l != r:
                raise ValueError("Can't copy sequence with size %d to array axis %d with dimension %d" % (r, i, l))
        n_elements = functools.reduce(operator.mul, lhs.shape, 1)
        _assign_kernel(lhs.ndim).forall(n_elements, stream=stream)(lhs, rhs)
        if synchronous:
            stream.synchronize()