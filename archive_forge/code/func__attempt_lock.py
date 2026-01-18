import os
import time
import yaml
from . import config, debug, errors, lock, osutils, ui, urlutils
from .decorators import only_raises
from .errors import (DirectoryNotEmpty, LockBreakMismatch, LockBroken,
from .i18n import gettext
from .osutils import format_delta, get_host_name, rand_chars
from .trace import mutter, note
from .transport import FileExists, NoSuchFile
def _attempt_lock(self):
    """Make the pending directory and attempt to rename into place.

        If the rename succeeds, we read back the info file to check that we
        really got the lock.

        If we fail to acquire the lock, this method is responsible for
        cleaning up the pending directory if possible.  (But it doesn't do
        that yet.)

        :returns: The nonce of the lock, if it was successfully acquired.

        :raises LockContention: If the lock is held by someone else.  The
            exception contains the info of the current holder of the lock.
        """
    self._trace('lock_write...')
    start_time = time.time()
    try:
        tmpname = self._create_pending_dir()
    except (errors.TransportError, PathError) as e:
        self._trace('... failed to create pending dir, %s', e)
        raise LockFailed(self, e)
    while True:
        try:
            self.transport.rename(tmpname, self._held_dir)
            break
        except (errors.TransportError, PathError, DirectoryNotEmpty, FileExists, ResourceBusy) as e:
            self._trace('... contention, %s', e)
            other_holder = self.peek()
            self._trace('other holder is %r' % other_holder)
            try:
                self._handle_lock_contention(other_holder)
            except BaseException:
                self._remove_pending_dir(tmpname)
                raise
        except Exception as e:
            self._trace('... lock failed, %s', e)
            self._remove_pending_dir(tmpname)
            raise
    info = self.peek()
    self._trace('after locking, info=%r', info)
    if info is None:
        raise LockFailed(self, 'lock was renamed into place, but now is missing!')
    if info.nonce != self.nonce:
        self._trace('rename succeeded, but lock is still held by someone else')
        raise LockContention(self)
    self._lock_held = True
    self._trace('... lock succeeded after %dms', (time.time() - start_time) * 1000)
    return self.nonce