from ..counted_lock import CountedLock
from ..errors import LockError, LockNotHeld, ReadOnlyError, TokenMismatch
from . import TestCase
def _assert_locked(self):
    if not self._lock_mode:
        raise LockError('{} is not locked'.format(self))