import wrapt
@wrapt.decorator
def _synced(method, self, args, kwargs):
    """Underlying synchronized wrapper."""
    with self._lock:
        return method(*args, **kwargs)