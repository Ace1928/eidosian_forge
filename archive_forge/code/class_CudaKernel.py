from array import array
import re
import ctypes
import numpy as np
from .base import _LIB, mx_uint, c_array, c_array_buf, c_str_array, check_call
from .base import c_str, CudaModuleHandle, CudaKernelHandle, numeric_types, string_types
from .ndarray import _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP, NDArray
class CudaKernel(object):
    """Constructs CUDA kernel. Should be created by `CudaModule.get_kernel`,
    not intended to be used by users.
    """

    def __init__(self, handle, name, is_ndarray, dtypes):
        self.handle = handle
        self._name = name
        self._is_ndarray = is_ndarray
        self._dtypes = [_DTYPE_MX_TO_NP[i] for i in dtypes]

    def __del__(self):
        check_call(_LIB.MXRtcCudaKernelFree(self.handle))

    def launch(self, args, ctx, grid_dims, block_dims, shared_mem=0):
        """Launch cuda kernel.

        Parameters
        ----------
        args : tuple of NDArray or numbers
            List of arguments for kernel. NDArrays are expected for pointer
            types (e.g. `float*`, `double*`) while numbers are expected for
            non-pointer types (e.g. `int`, `float`).
        ctx : Context
            The context to launch kernel on. Must be GPU context.
        grid_dims : tuple of 3 integers
            Grid dimensions for CUDA kernel.
        block_dims : tuple of 3 integers
            Block dimensions for CUDA kernel.
        shared_mem : integer, optional
            Size of dynamically allocated shared memory. Defaults to 0.
        """
        assert ctx.device_type == 'gpu', 'Cuda kernel can only be launched on GPU'
        assert len(grid_dims) == 3, 'grid_dims must be a tuple of 3 integers'
        assert len(block_dims) == 3, 'grid_dims must be a tuple of 3 integers'
        assert len(args) == len(self._dtypes), 'CudaKernel(%s) expects %d arguments but got %d' % (self._name, len(self._dtypes), len(args))
        void_args = []
        ref_holder = []
        for i, (arg, is_nd, dtype) in enumerate(zip(args, self._is_ndarray, self._dtypes)):
            if is_nd:
                assert isinstance(arg, NDArray), 'The %d-th argument is expected to be a NDArray but got %s' % (i, type(arg))
                void_args.append(arg.handle)
            else:
                assert isinstance(arg, numeric_types), 'The %d-th argument is expected to be a number, but got %s' % (i, type(arg))
                ref_holder.append(np.array(arg, dtype=dtype))
                void_args.append(ref_holder[-1].ctypes.data_as(ctypes.c_void_p))
        check_call(_LIB.MXRtcCudaKernelCall(self.handle, ctx.device_id, c_array(ctypes.c_void_p, void_args), mx_uint(grid_dims[0]), mx_uint(grid_dims[1]), mx_uint(grid_dims[2]), mx_uint(block_dims[0]), mx_uint(block_dims[1]), mx_uint(block_dims[2]), mx_uint(shared_mem)))