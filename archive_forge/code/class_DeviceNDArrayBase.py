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
class DeviceNDArrayBase(_devicearray.DeviceArray):
    """A on GPU NDArray representation
    """
    __cuda_memory__ = True
    __cuda_ndarray__ = True

    def __init__(self, shape, strides, dtype, stream=0, gpu_data=None):
        """
        Args
        ----

        shape
            array shape.
        strides
            array strides.
        dtype
            data type as np.dtype coercible object.
        stream
            cuda stream.
        gpu_data
            user provided device memory for the ndarray data buffer
        """
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(strides, int):
            strides = (strides,)
        dtype = np.dtype(dtype)
        self.ndim = len(shape)
        if len(strides) != self.ndim:
            raise ValueError('strides not match ndim')
        self._dummy = dummyarray.Array.from_desc(0, shape, strides, dtype.itemsize)
        self.shape = tuple(shape)
        self.strides = tuple(strides)
        self.dtype = dtype
        self.size = int(functools.reduce(operator.mul, self.shape, 1))
        if self.size > 0:
            if gpu_data is None:
                self.alloc_size = _driver.memory_size_from_info(self.shape, self.strides, self.dtype.itemsize)
                gpu_data = devices.get_context().memalloc(self.alloc_size)
            else:
                self.alloc_size = _driver.device_memory_size(gpu_data)
        else:
            if _driver.USE_NV_BINDING:
                null = _driver.binding.CUdeviceptr(0)
            else:
                null = c_void_p(0)
            gpu_data = _driver.MemoryPointer(context=devices.get_context(), pointer=null, size=0)
            self.alloc_size = 0
        self.gpu_data = gpu_data
        self.stream = stream

    @property
    def __cuda_array_interface__(self):
        if _driver.USE_NV_BINDING:
            if self.device_ctypes_pointer is not None:
                ptr = int(self.device_ctypes_pointer)
            else:
                ptr = 0
        elif self.device_ctypes_pointer.value is not None:
            ptr = self.device_ctypes_pointer.value
        else:
            ptr = 0
        return {'shape': tuple(self.shape), 'strides': None if is_contiguous(self) else tuple(self.strides), 'data': (ptr, False), 'typestr': self.dtype.str, 'stream': int(self.stream) if self.stream != 0 else None, 'version': 3}

    def bind(self, stream=0):
        """Bind a CUDA stream to this object so that all subsequent operation
        on this array defaults to the given stream.
        """
        clone = copy.copy(self)
        clone.stream = stream
        return clone

    @property
    def T(self):
        return self.transpose()

    def transpose(self, axes=None):
        if axes and tuple(axes) == tuple(range(self.ndim)):
            return self
        elif self.ndim != 2:
            msg = "transposing a non-2D DeviceNDArray isn't supported"
            raise NotImplementedError(msg)
        elif axes is not None and set(axes) != set(range(self.ndim)):
            raise ValueError('invalid axes list %r' % (axes,))
        else:
            from numba.cuda.kernels.transpose import transpose
            return transpose(self)

    def _default_stream(self, stream):
        return self.stream if not stream else stream

    @property
    def _numba_type_(self):
        """
        Magic attribute expected by Numba to get the numba type that
        represents this object.
        """
        broadcast = 0 in self.strides
        if self.flags['C_CONTIGUOUS'] and (not broadcast):
            layout = 'C'
        elif self.flags['F_CONTIGUOUS'] and (not broadcast):
            layout = 'F'
        else:
            layout = 'A'
        dtype = numpy_support.from_dtype(self.dtype)
        return types.Array(dtype, self.ndim, layout)

    @property
    def device_ctypes_pointer(self):
        """Returns the ctypes pointer to the GPU data buffer
        """
        if self.gpu_data is None:
            if _driver.USE_NV_BINDING:
                return _driver.binding.CUdeviceptr(0)
            else:
                return c_void_p(0)
        else:
            return self.gpu_data.device_ctypes_pointer

    @devices.require_context
    def copy_to_device(self, ary, stream=0):
        """Copy `ary` to `self`.

        If `ary` is a CUDA memory, perform a device-to-device transfer.
        Otherwise, perform a a host-to-device transfer.
        """
        if ary.size == 0:
            return
        sentry_contiguous(self)
        stream = self._default_stream(stream)
        self_core, ary_core = (array_core(self), array_core(ary))
        if _driver.is_device_memory(ary):
            sentry_contiguous(ary)
            check_array_compatibility(self_core, ary_core)
            _driver.device_to_device(self, ary, self.alloc_size, stream=stream)
        else:
            ary_core = np.array(ary_core, order='C' if self_core.flags['C_CONTIGUOUS'] else 'F', subok=True, copy=not ary_core.flags['WRITEABLE'])
            check_array_compatibility(self_core, ary_core)
            _driver.host_to_device(self, ary_core, self.alloc_size, stream=stream)

    @devices.require_context
    def copy_to_host(self, ary=None, stream=0):
        """Copy ``self`` to ``ary`` or create a new Numpy ndarray
        if ``ary`` is ``None``.

        If a CUDA ``stream`` is given, then the transfer will be made
        asynchronously as part as the given stream.  Otherwise, the transfer is
        synchronous: the function returns after the copy is finished.

        Always returns the host array.

        Example::

            import numpy as np
            from numba import cuda

            arr = np.arange(1000)
            d_arr = cuda.to_device(arr)

            my_kernel[100, 100](d_arr)

            result_array = d_arr.copy_to_host()
        """
        if any((s < 0 for s in self.strides)):
            msg = 'D->H copy not implemented for negative strides: {}'
            raise NotImplementedError(msg.format(self.strides))
        assert self.alloc_size >= 0, 'Negative memory size'
        stream = self._default_stream(stream)
        if ary is None:
            hostary = np.empty(shape=self.alloc_size, dtype=np.byte)
        else:
            check_array_compatibility(self, ary)
            hostary = ary
        if self.alloc_size != 0:
            _driver.device_to_host(hostary, self, self.alloc_size, stream=stream)
        if ary is None:
            if self.size == 0:
                hostary = np.ndarray(shape=self.shape, dtype=self.dtype, buffer=hostary)
            else:
                hostary = np.ndarray(shape=self.shape, dtype=self.dtype, strides=self.strides, buffer=hostary)
        return hostary

    def split(self, section, stream=0):
        """Split the array into equal partition of the `section` size.
        If the array cannot be equally divided, the last section will be
        smaller.
        """
        stream = self._default_stream(stream)
        if self.ndim != 1:
            raise ValueError('only support 1d array')
        if self.strides[0] != self.dtype.itemsize:
            raise ValueError('only support unit stride')
        nsect = int(math.ceil(float(self.size) / section))
        strides = self.strides
        itemsize = self.dtype.itemsize
        for i in range(nsect):
            begin = i * section
            end = min(begin + section, self.size)
            shape = (end - begin,)
            gpu_data = self.gpu_data.view(begin * itemsize, end * itemsize)
            yield DeviceNDArray(shape, strides, dtype=self.dtype, stream=stream, gpu_data=gpu_data)

    def as_cuda_arg(self):
        """Returns a device memory object that is used as the argument.
        """
        return self.gpu_data

    def get_ipc_handle(self):
        """
        Returns a *IpcArrayHandle* object that is safe to serialize and transfer
        to another process to share the local allocation.

        Note: this feature is only available on Linux.
        """
        ipch = devices.get_context().get_ipc_handle(self.gpu_data)
        desc = dict(shape=self.shape, strides=self.strides, dtype=self.dtype)
        return IpcArrayHandle(ipc_handle=ipch, array_desc=desc)

    def squeeze(self, axis=None, stream=0):
        """
        Remove axes of size one from the array shape.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Subset of dimensions to remove. A `ValueError` is raised if an axis
            with size greater than one is selected. If `None`, all axes with
            size one are removed.
        stream : cuda stream or 0, optional
            Default stream for the returned view of the array.

        Returns
        -------
        DeviceNDArray
            Squeezed view into the array.

        """
        new_dummy, _ = self._dummy.squeeze(axis=axis)
        return DeviceNDArray(shape=new_dummy.shape, strides=new_dummy.strides, dtype=self.dtype, stream=self._default_stream(stream), gpu_data=self.gpu_data)

    def view(self, dtype):
        """Returns a new object by reinterpretting the dtype without making a
        copy of the data.
        """
        dtype = np.dtype(dtype)
        shape = list(self.shape)
        strides = list(self.strides)
        if self.dtype.itemsize != dtype.itemsize:
            if not self.is_c_contiguous():
                raise ValueError('To change to a dtype of a different size, the array must be C-contiguous')
            shape[-1], rem = divmod(shape[-1] * self.dtype.itemsize, dtype.itemsize)
            if rem != 0:
                raise ValueError('When changing to a larger dtype, its size must be a divisor of the total size in bytes of the last axis of the array.')
            strides[-1] = dtype.itemsize
        return DeviceNDArray(shape=shape, strides=strides, dtype=dtype, stream=self.stream, gpu_data=self.gpu_data)

    @property
    def nbytes(self):
        return self.dtype.itemsize * self.size