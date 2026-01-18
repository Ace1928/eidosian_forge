from __future__ import absolute_import, division
import time
import os
from . import LockBase, NotLocked, NotMyLock, LockTimeout, AlreadyLocked
def _who_is_locking(self):
    cursor = self.connection.cursor()
    cursor.execute('select unique_name from locks  where lock_file = ?', (self.lock_file,))
    return cursor.fetchone()[0]