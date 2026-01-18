import threading
import functools
import numba.core.event as ev
@functools.wraps(func)
def _acquire_compile_lock(*args, **kwargs):
    with self:
        return func(*args, **kwargs)