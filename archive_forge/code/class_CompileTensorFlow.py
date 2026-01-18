import functools
from .autoray import (
from . import lazy
class CompileTensorFlow:
    """ """

    def __init__(self, fn, **kwargs):
        self._fn = fn
        kwargs.setdefault('autograph', False)
        self._jit_fn = None
        self._jit_kwargs = kwargs

    def setup(self):
        import tensorflow as tf
        self._jit_fn = tf.function(**self._jit_kwargs)(self._fn)
        self._fn = None

    def __call__(self, *args, array_backend=None, **kwargs):
        if self._jit_fn is None:
            self.setup()
        out = self._jit_fn(*args, **kwargs)
        if array_backend != 'tensorflow':
            out = do('asarray', out, like=array_backend)
        return out