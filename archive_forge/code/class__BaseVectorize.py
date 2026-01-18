import inspect
from numba.np.ufunc import _internal
from numba.np.ufunc.parallel import ParallelUFuncBuilder, ParallelGUFuncBuilder
from numba.core.registry import DelayedRegistry
from numba.np.ufunc import dufunc
from numba.np.ufunc import gufunc
class _BaseVectorize(object):

    @classmethod
    def get_identity(cls, kwargs):
        return kwargs.pop('identity', None)

    @classmethod
    def get_cache(cls, kwargs):
        return kwargs.pop('cache', False)

    @classmethod
    def get_writable_args(cls, kwargs):
        return kwargs.pop('writable_args', ())

    @classmethod
    def get_target_implementation(cls, kwargs):
        target = kwargs.pop('target', 'cpu')
        try:
            return cls.target_registry[target]
        except KeyError:
            raise ValueError('Unsupported target: %s' % target)