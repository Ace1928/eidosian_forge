import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def _get_errno(self):
    return self._backend.get_errno()