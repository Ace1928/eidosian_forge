import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def _pointer_to(self, ctype):
    with self._lock:
        return model.pointer_cache(self, ctype)