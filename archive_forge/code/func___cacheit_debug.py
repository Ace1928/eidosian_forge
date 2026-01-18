from importlib import import_module
from typing import Callable
from functools import lru_cache, wraps
def __cacheit_debug(maxsize):
    """cacheit + code to check cache consistency"""

    def func_wrapper(func):
        cfunc = __cacheit(maxsize)(func)

        @wraps(func)
        def wrapper(*args, **kw_args):
            r1 = func(*args, **kw_args)
            r2 = cfunc(*args, **kw_args)
            (hash(r1), hash(r2))
            if r1 != r2:
                raise RuntimeError('Returned values are not the same')
            return r1
        return wrapper
    return func_wrapper