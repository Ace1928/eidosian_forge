import functools
import threading
from contextlib import contextmanager
from .driver import driver, USE_NV_BINDING
@functools.wraps(fn)
def _require_cuda_context(*args, **kws):
    with _runtime.ensure_context():
        return fn(*args, **kws)