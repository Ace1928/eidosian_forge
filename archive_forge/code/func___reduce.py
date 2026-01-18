from numba import cuda
from numpy import array as np_array
from numba.np.ufunc import deviceufunc
from numba.np.ufunc.deviceufunc import (UFuncMechanism, GeneralizedUFunc,
def __reduce(self, mem, gpu_mems, stream):
    n = mem.shape[0]
    if n % 2 != 0:
        fatcut, thincut = mem.split(n - 1)
        gpu_mems.append(fatcut)
        gpu_mems.append(thincut)
        out = self.__reduce(fatcut, gpu_mems, stream)
        gpu_mems.append(out)
        return self(out, thincut, out=out, stream=stream)
    else:
        left, right = mem.split(n // 2)
        gpu_mems.append(left)
        gpu_mems.append(right)
        self(left, right, out=left, stream=stream)
        if n // 2 > 1:
            return self.__reduce(left, gpu_mems, stream)
        else:
            return left