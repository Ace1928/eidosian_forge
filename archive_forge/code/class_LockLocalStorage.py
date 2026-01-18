import os
import time
import errno
import shutil
import tempfile
import threading
from hashlib import sha256
from libcloud.utils.py3 import u, relpath
from libcloud.common.base import Connection
from libcloud.utils.files import read_in_chunks, exhaust_iterator
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
class LockLocalStorage:
    """
    A class which locks a local path which is being updated. To correctly handle all the scenarios
    use a thread based and IPC based lock.
    """

    def __init__(self, path, timeout=5):
        self.path = path
        self.lock_acquire_timeout = timeout
        self.ipc_lock_path = os.path.join(tempfile.gettempdir(), '%s.lock' % sha256(path.encode('utf-8')).hexdigest())
        self.thread_lock = threading.Lock()
        self.ipc_lock = fasteners.InterProcessLock(self.ipc_lock_path)

    def __enter__(self):
        lock_acquire_timeout = self.lock_acquire_timeout
        start_time = int(time.time())
        end_time = start_time + lock_acquire_timeout
        while int(time.time()) < end_time:
            success = self.thread_lock.acquire(blocking=False)
            if success:
                break
        if not success:
            raise LibcloudError('Failed to acquire thread lock for path %s in %s seconds' % (self.path, lock_acquire_timeout))
        success = self.ipc_lock.acquire(blocking=True, timeout=lock_acquire_timeout)
        if not success:
            raise LibcloudError('Failed to acquire IPC lock (%s) for path %s in %s seconds' % (self.ipc_lock_path, self.path, lock_acquire_timeout))

    def __exit__(self, type, value, traceback):
        if self.thread_lock.locked():
            self.thread_lock.release()
        if self.ipc_lock.exists():
            self.ipc_lock.release()
        if value is not None:
            raise value