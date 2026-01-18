import collections
import contextlib
import functools
from concurrent import futures
from concurrent.futures import _base
import futurist
from futurist import _utils
def _ensure_eventlet(func):
    """Decorator that verifies we have the needed eventlet components."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _utils.EVENTLET_AVAILABLE or greenthreading is None:
            raise RuntimeError('Eventlet is needed to wait on green futures')
        return func(*args, **kwargs)
    return wrapper