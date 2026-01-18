import ctypes
from .base import _LIB, check_call
class _BulkScope(object):
    """Scope object for bulk execution."""

    def __init__(self, size):
        self._size = size
        self._old_size = None

    def __enter__(self):
        self._old_size = set_bulk_size(self._size)
        return self

    def __exit__(self, ptype, value, trace):
        set_bulk_size(self._old_size)