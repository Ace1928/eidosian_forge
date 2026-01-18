import functools
import inspect
import types
from typing import Type, Callable
class _WrappedMethod:
    """Descriptor which calls its two arguments in succession, correctly handling instance- and
    class-method calls.

    It is intended that this class will replace the attribute that ``inner`` previously was on a
    class or instance.  When accessed as that attribute, this descriptor will behave it is the same
    function call, but with the ``function`` called before or after.
    """
    __slots__ = ('_method_decorator', '_method_has_get', '_method', '_before', '_after')

    def __init__(self, method, before=None, after=None):
        if isinstance(method, (classmethod, staticmethod)):
            self._method_decorator = type(method)
        elif isinstance(method, type(self)):
            self._method_decorator = method._method_decorator
        elif getattr(method, '__name__', None) in _MAGIC_STATICMETHODS:
            self._method_decorator = staticmethod
        elif getattr(method, '__name__', None) in _MAGIC_CLASSMETHODS:
            self._method_decorator = classmethod
        else:
            self._method_decorator = _lift_to_method
        before = (self._method_decorator(before),) if before is not None else ()
        after = (self._method_decorator(after),) if after is not None else ()
        if isinstance(method, type(self)):
            self._method = method._method
            self._before = before + method._before
            self._after = method._after + after
        else:
            self._before = before
            self._after = after
            self._method = method
        self._method_has_get = hasattr(self._method, '__get__')

    def __get__(self, obj, objtype=None):
        method = self._method.__get__(obj, objtype) if self._method_has_get else self._method

        @functools.wraps(method)
        def out(*args, **kwargs):
            for callback in self._before:
                callback.__get__(obj, objtype)(*args, **kwargs)
            retval = method(*args, **kwargs)
            for callback in self._after:
                callback.__get__(obj, objtype)(*args, **kwargs)
            return retval
        return out