from abc import ABC, abstractmethod
from numba.core.registry import DelayedRegistry, CPUDispatcher
from numba.core.decorators import jit
from numba.core.errors import InternalTargetMismatchError, NumbaValueError
from threading import local as tls
class target_override(object):
    """Context manager to temporarily override the current target with that
       prescribed."""

    def __init__(self, name):
        self._orig_target = getattr(_active_context, 'target', _active_context_default)
        self.target = name

    def __enter__(self):
        _active_context.target = self.target

    def __exit__(self, ty, val, tb):
        _active_context.target = self._orig_target