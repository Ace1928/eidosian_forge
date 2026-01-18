import os
import sys
import time
import shutil
import platform
import tempfile
import unittest
import multiprocessing
from libcloud.utils.files import exhaust_iterator
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container
from libcloud.storage.types import (
class PickleableAcquireLockInSubprocess:

    def __call__(self, pid, success):
        lock = LockLocalStorage('/tmp/c', timeout=0.5)
        if pid == 1:
            with lock:
                time.sleep(2.5)
            success.value = 1
        elif pid == 2:
            expected_msg = 'Failed to acquire IPC lock'
            try:
                lock.__enter__()
            except LibcloudError as e:
                assert expected_msg in str(e)
                success.value = 1
        else:
            raise ValueError('Invalid pid')