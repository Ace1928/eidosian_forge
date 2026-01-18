from typing import Optional, Type, Union
from . import counted_lock, errors, lock, transactions, urlutils
from .decorators import only_raises
from .transport import Transport
class TransportLock:
    """Locking method which uses transport-dependent locks.

    On the local filesystem these transform into OS-managed locks.

    These do not guard against concurrent access via different
    transports.

    This is suitable for use only in WorkingTrees (which are at present
    always local).
    """

    def __init__(self, transport: Transport, escaped_name: str, file_modebits, dir_modebits):
        self._transport = transport
        self._escaped_name = escaped_name
        self._file_modebits = file_modebits
        self._dir_modebits = dir_modebits

    def break_lock(self):
        raise NotImplementedError(self.break_lock)

    def leave_in_place(self):
        raise NotImplementedError(self.leave_in_place)

    def dont_leave_in_place(self):
        raise NotImplementedError(self.dont_leave_in_place)

    def lock_write(self, token=None):
        if token is not None:
            raise errors.TokenLockingNotSupported(self)
        self._lock = self._transport.lock_write(self._escaped_name)

    def lock_read(self):
        self._lock = self._transport.lock_read(self._escaped_name)

    def unlock(self):
        self._lock.unlock()
        self._lock = None

    def peek(self):
        raise NotImplementedError()

    def create(self, mode=None):
        """Create lock mechanism"""
        self._transport.put_bytes(self._escaped_name, b'', mode=self._file_modebits)

    def validate_token(self, token):
        if token is not None:
            raise errors.TokenLockingNotSupported(self)