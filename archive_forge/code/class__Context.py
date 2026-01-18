import threading
class _Context(object):
    """Context manager helper for `GroupLock`."""
    __slots__ = ['_lock', '_group_id']

    def __init__(self, lock, group_id):
        self._lock = lock
        self._group_id = group_id

    def __enter__(self):
        self._lock.acquire(self._group_id)

    def __exit__(self, type_arg, value_arg, traceback_arg):
        del type_arg, value_arg, traceback_arg
        self._lock.release(self._group_id)