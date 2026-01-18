import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def _get_cached_btype(self, type):
    assert self._lock.acquire(False) is False
    try:
        BType = self._cached_btypes[type]
    except KeyError:
        finishlist = []
        BType = type.get_cached_btype(self, finishlist)
        for type in finishlist:
            type.finish_backend_type(self, finishlist)
    return BType