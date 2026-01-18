from ..counted_lock import CountedLock
from ..errors import LockError, LockNotHeld, ReadOnlyError, TokenMismatch
from . import TestCase
class DummyLock:
    """Lock that just records what's been done to it."""

    def __init__(self):
        self._calls = []
        self._lock_mode = None

    def is_locked(self):
        return self._lock_mode is not None

    def lock_read(self):
        self._assert_not_locked()
        self._lock_mode = 'r'
        self._calls.append('lock_read')

    def lock_write(self, token=None):
        if token is not None:
            if token == 'token':
                return 'token'
            else:
                raise TokenMismatch()
        self._assert_not_locked()
        self._lock_mode = 'w'
        self._calls.append('lock_write')
        return 'token'

    def unlock(self):
        self._assert_locked()
        self._lock_mode = None
        self._calls.append('unlock')

    def break_lock(self):
        self._lock_mode = None
        self._calls.append('break')

    def _assert_locked(self):
        if not self._lock_mode:
            raise LockError('{} is not locked'.format(self))

    def _assert_not_locked(self):
        if self._lock_mode:
            raise LockError('%s is already locked in mode %r' % (self, self._lock_mode))

    def validate_token(self, token):
        if token == 'token':
            return 'token'
        elif token is None:
            return
        else:
            raise TokenMismatch(token, 'token')