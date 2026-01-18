import base64
import json
import logging
import os
import threading
import fasteners
from six import iteritems
from oauth2client import _helpers
from oauth2client import client
class _MultiprocessStorageBackend(object):
    """Thread-local backend for multiprocess storage.

    Each process has only one instance of this backend per file. All threads
    share a single instance of this backend. This ensures that all threads
    use the same thread lock and process lock when accessing the file.
    """

    def __init__(self, filename):
        self._file = None
        self._filename = filename
        self._process_lock = fasteners.InterProcessLock('{0}.lock'.format(filename))
        self._thread_lock = threading.Lock()
        self._read_only = False
        self._credentials = {}

    def _load_credentials(self):
        """(Re-)loads the credentials from the file."""
        if not self._file:
            return
        loaded_credentials = _load_credentials_file(self._file)
        self._credentials.update(loaded_credentials)
        logger.debug('Read credential file')

    def _write_credentials(self):
        if self._read_only:
            logger.debug('In read-only mode, not writing credentials.')
            return
        _write_credentials_file(self._file, self._credentials)
        logger.debug('Wrote credential file {0}.'.format(self._filename))

    def acquire_lock(self):
        self._thread_lock.acquire()
        locked = self._process_lock.acquire(timeout=INTERPROCESS_LOCK_DEADLINE)
        if locked:
            _create_file_if_needed(self._filename)
            self._file = open(self._filename, 'r+')
            self._read_only = False
        else:
            logger.warn('Failed to obtain interprocess lock for credentials. If a credential is being refreshed, other processes may not see the updated access token and refresh as well.')
            if os.path.exists(self._filename):
                self._file = open(self._filename, 'r')
            else:
                self._file = None
            self._read_only = True
        self._load_credentials()

    def release_lock(self):
        if self._file is not None:
            self._file.close()
            self._file = None
        if not self._read_only:
            self._process_lock.release()
        self._thread_lock.release()

    def _refresh_predicate(self, credentials):
        if credentials is None:
            return True
        elif credentials.invalid:
            return True
        elif credentials.access_token_expired:
            return True
        else:
            return False

    def locked_get(self, key):
        credentials = self._credentials.get(key, None)
        if self._refresh_predicate(credentials):
            self._load_credentials()
            credentials = self._credentials.get(key, None)
        return credentials

    def locked_put(self, key, credentials):
        self._load_credentials()
        self._credentials[key] = credentials
        self._write_credentials()

    def locked_delete(self, key):
        self._load_credentials()
        self._credentials.pop(key, None)
        self._write_credentials()