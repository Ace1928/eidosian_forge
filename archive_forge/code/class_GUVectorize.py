import inspect
from numba.np.ufunc import _internal
from numba.np.ufunc.parallel import ParallelUFuncBuilder, ParallelGUFuncBuilder
from numba.core.registry import DelayedRegistry
from numba.np.ufunc import dufunc
from numba.np.ufunc import gufunc
class GUVectorize(_BaseVectorize):
    target_registry = DelayedRegistry({'cpu': gufunc.GUFunc, 'parallel': ParallelGUFuncBuilder})

    def __new__(cls, func, signature, **kws):
        identity = cls.get_identity(kws)
        cache = cls.get_cache(kws)
        imp = cls.get_target_implementation(kws)
        writable_args = cls.get_writable_args(kws)
        if imp is gufunc.GUFunc:
            is_dyn = kws.pop('is_dynamic', False)
            return imp(func, signature, identity=identity, cache=cache, is_dynamic=is_dyn, targetoptions=kws, writable_args=writable_args)
        else:
            return imp(func, signature, identity=identity, cache=cache, targetoptions=kws, writable_args=writable_args)