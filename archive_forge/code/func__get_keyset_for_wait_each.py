from eventlet.event import Event
from eventlet import greenthread
import collections
def _get_keyset_for_wait_each(self, keys):
    """
        wait_each(), wait_each_success() and wait_each_exception() promise
        that if you pass an iterable of keys, the method will wait for results
        from those keys -- but if you omit the keys argument, the method will
        wait for results from all known keys. This helper implements that
        distinction, returning a set() of the relevant keys.
        """
    if keys is not _MISSING:
        return set(keys)
    else:
        return set(self.coros.keys()) | set(self.values.keys())