from importlib import import_module
from typing import Callable
from functools import lru_cache, wraps
def __cacheit(maxsize):
    """caching decorator.

        important: the result of cached function must be *immutable*


        Examples
        ========

        >>> from sympy import cacheit
        >>> @cacheit
        ... def f(a, b):
        ...    return a+b

        >>> @cacheit
        ... def f(a, b): # noqa: F811
        ...    return [a, b] # <-- WRONG, returns mutable object

        to force cacheit to check returned results mutability and consistency,
        set environment variable SYMPY_USE_CACHE to 'debug'
    """

    def func_wrapper(func):
        cfunc = lru_cache(maxsize, typed=True)(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                retval = cfunc(*args, **kwargs)
            except TypeError as e:
                if not e.args or not e.args[0].startswith('unhashable type:'):
                    raise
                retval = func(*args, **kwargs)
            return retval
        wrapper.cache_info = cfunc.cache_info
        wrapper.cache_clear = cfunc.cache_clear
        CACHE.append(wrapper)
        return wrapper
    return func_wrapper