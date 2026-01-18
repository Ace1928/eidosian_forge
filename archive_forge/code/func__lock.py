import errno
import json
import logging
import os
import threading
from oauth2client import client
from oauth2client import util
from oauth2client.contrib import locked_file
def _lock(self):
    """Lock the entire multistore."""
    self._thread_lock.acquire()
    try:
        self._file.open_and_lock()
    except (IOError, OSError) as e:
        if e.errno == errno.ENOSYS:
            logger.warn('File system does not support locking the credentials file.')
        elif e.errno == errno.ENOLCK:
            logger.warn('File system is out of resources for writing the credentials file (is your disk full?).')
        elif e.errno == errno.EDEADLK:
            logger.warn('Lock contention on multistore file, opening in read-only mode.')
        elif e.errno == errno.EACCES:
            logger.warn('Cannot access credentials file.')
        else:
            raise
    if not self._file.is_locked():
        self._read_only = True
        if self._warn_on_readonly:
            logger.warn('The credentials file (%s) is not writable. Opening in read-only mode. Any refreshed credentials will only be valid for this run.', self._file.filename())
    if os.path.getsize(self._file.filename()) == 0:
        logger.debug('Initializing empty multistore file')
        self._data = {}
        self._write()
    elif not self._read_only or self._data is None:
        self._refresh_data_cache()