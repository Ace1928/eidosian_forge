from numba import cuda
from numpy import array as np_array
from numba.np.ufunc import deviceufunc
from numba.np.ufunc.deviceufunc import (UFuncMechanism, GeneralizedUFunc,
class CUDAUFuncMechanism(UFuncMechanism):
    """
    Provide CUDA specialization
    """
    DEFAULT_STREAM = 0

    def launch(self, func, count, stream, args):
        func.forall(count, stream=stream)(*args)

    def is_device_array(self, obj):
        return cuda.is_cuda_array(obj)

    def as_device_array(self, obj):
        if cuda.cudadrv.devicearray.is_cuda_ndarray(obj):
            return obj
        return cuda.as_cuda_array(obj)

    def to_device(self, hostary, stream):
        return cuda.to_device(hostary, stream=stream)

    def to_host(self, devary, stream):
        return devary.copy_to_host(stream=stream)

    def allocate_device_array(self, shape, dtype, stream):
        return cuda.device_array(shape=shape, dtype=dtype, stream=stream)

    def broadcast_device(self, ary, shape):
        ax_differs = [ax for ax in range(len(shape)) if ax >= ary.ndim or ary.shape[ax] != shape[ax]]
        missingdim = len(shape) - len(ary.shape)
        strides = [0] * missingdim + list(ary.strides)
        for ax in ax_differs:
            strides[ax] = 0
        return cuda.cudadrv.devicearray.DeviceNDArray(shape=shape, strides=strides, dtype=ary.dtype, gpu_data=ary.gpu_data)