import inspect
from numba.np.ufunc import _internal
from numba.np.ufunc.parallel import ParallelUFuncBuilder, ParallelGUFuncBuilder
from numba.core.registry import DelayedRegistry
from numba.np.ufunc import dufunc
from numba.np.ufunc import gufunc
class Vectorize(_BaseVectorize):
    target_registry = DelayedRegistry({'cpu': dufunc.DUFunc, 'parallel': ParallelUFuncBuilder})

    def __new__(cls, func, **kws):
        identity = cls.get_identity(kws)
        cache = cls.get_cache(kws)
        imp = cls.get_target_implementation(kws)
        return imp(func, identity=identity, cache=cache, targetoptions=kws)