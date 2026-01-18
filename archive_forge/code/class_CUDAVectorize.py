from numba import cuda
from numpy import array as np_array
from numba.np.ufunc import deviceufunc
from numba.np.ufunc.deviceufunc import (UFuncMechanism, GeneralizedUFunc,
class CUDAVectorize(deviceufunc.DeviceVectorize):

    def _compile_core(self, sig):
        cudevfn = cuda.jit(sig, device=True, inline=True)(self.pyfunc)
        return (cudevfn, cudevfn.overloads[sig.args].signature.return_type)

    def _get_globals(self, corefn):
        glbl = self.pyfunc.__globals__.copy()
        glbl.update({'__cuda__': cuda, '__core__': corefn})
        return glbl

    def _compile_kernel(self, fnobj, sig):
        return cuda.jit(fnobj)

    def build_ufunc(self):
        return CUDAUFuncDispatcher(self.kernelmap, self.pyfunc)

    @property
    def _kernel_template(self):
        return vectorizer_stager_source