import sys
import abc
import io
import copyreg
import pickle
from numba import cloudpickle
from llvmlite import ir
class PickleCallableByPath:
    """Wrap a callable object to be pickled by path to workaround limitation
    in pickling due to non-pickleable objects in function non-locals.

    Note:
    - Do not use this as a decorator.
    - Wrapped object must be a global that exist in its parent module and it
      can be imported by `from the_module import the_object`.

    Usage:

    >>> def my_fn(x):
    >>>     ...
    >>> wrapped_fn = PickleCallableByPath(my_fn)
    >>> # refer to `wrapped_fn` instead of `my_fn`
    """

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __reduce__(self):
        return (type(self)._rebuild, (self._fn.__module__, self._fn.__name__))

    @classmethod
    def _rebuild(cls, modname, fn_path):
        return cls(getattr(sys.modules[modname], fn_path))