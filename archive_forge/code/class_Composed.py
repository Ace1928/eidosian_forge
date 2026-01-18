import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
class Composed:
    """Compose an ``autoray.do`` using function. See the main wrapper
    ``compose``.
    """

    def __init__(self, fn, name=None):
        self._default_fn = fn
        if name is None:
            name = fn.__name__
        self._name = name
        self._supply_backend = 'backend' in signature(fn).parameters
        _COMPOSED_FUNCTION_GENERATORS[self._name] = self.make_function

    def register(self, backend, fn=None):
        """Register a different implementation for ``backend``."""
        if fn is not None:
            register_function(backend, self._name, fn)
        else:

            def wrapper(fn):
                register_function(backend, self._name, fn)
                return fn
            return wrapper

    def make_function(self, backend):
        """Make a new function for the specific ``backend``."""
        if self._supply_backend:
            fn = functools.wraps(self._default_fn)(functools.partial(self._default_fn, backend=backend))
        else:
            fn = self._default_fn
        self.register(backend, fn)
        return fn

    def __call__(self, *args, like=None, **kwargs):
        backend = choose_backend(self._name, *args, like=like, **kwargs)
        fn = get_lib_fn(backend, self._name)
        return fn(*args, **kwargs)

    def __repr__(self):
        return f"Composed('{self._name}')"