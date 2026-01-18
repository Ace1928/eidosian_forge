from ..counted_lock import CountedLock
from ..errors import LockError, LockNotHeld, ReadOnlyError, TokenMismatch
from . import TestCase
def _assert_not_locked(self):
    if self._lock_mode:
        raise LockError('%s is already locked in mode %r' % (self, self._lock_mode))