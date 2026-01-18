from numba import cuda
from numpy import array as np_array
from numba.np.ufunc import deviceufunc
from numba.np.ufunc.deviceufunc import (UFuncMechanism, GeneralizedUFunc,
class _CUDAGUFuncCallSteps(GUFuncCallSteps):
    __slots__ = ['_stream']

    def __init__(self, nin, nout, args, kwargs):
        super().__init__(nin, nout, args, kwargs)
        self._stream = kwargs.get('stream', 0)

    def is_device_array(self, obj):
        return cuda.is_cuda_array(obj)

    def as_device_array(self, obj):
        if cuda.cudadrv.devicearray.is_cuda_ndarray(obj):
            return obj
        return cuda.as_cuda_array(obj)

    def to_device(self, hostary):
        return cuda.to_device(hostary, stream=self._stream)

    def to_host(self, devary, hostary):
        out = devary.copy_to_host(hostary, stream=self._stream)
        return out

    def allocate_device_array(self, shape, dtype):
        return cuda.device_array(shape=shape, dtype=dtype, stream=self._stream)

    def launch_kernel(self, kernel, nelem, args):
        kernel.forall(nelem, stream=self._stream)(*args)