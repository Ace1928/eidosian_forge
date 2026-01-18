import functools
from .autoray import (
from . import lazy
class CompileJax:
    """ """

    def __init__(self, fn, enable_x64=None, platform_name=None, **kwargs):
        self._fn = fn
        self._enable_x64 = enable_x64
        self._platform_name = platform_name
        self._jit_fn = None
        self._jit_kwargs = kwargs

    def setup(self):
        import jax
        if self._enable_x64 is not None:
            import jax
            jax.config.update('jax_enable_x64', self._enable_x64)
        if self._platform_name is not None:
            import jax
            jax.config.update('jax_platform_name', self._platform_name)
        self._jit_fn = jax.jit(self._fn, **self._jit_kwargs)
        self._fn = None

    def __call__(self, *args, array_backend=None, **kwargs):
        if self._jit_fn is None:
            self.setup()
        out = self._jit_fn(*args, **kwargs)
        if array_backend != 'jax':
            out = do('asarray', out, like=array_backend)
        return out