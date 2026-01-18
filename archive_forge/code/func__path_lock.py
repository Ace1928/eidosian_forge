import contextlib
import errno
import io
import os
import shutil
import cachetools
import fasteners
from oslo_serialization import jsonutils
from oslo_utils import fileutils
from taskflow import exceptions as exc
from taskflow.persistence import path_based
from taskflow.utils import misc
@contextlib.contextmanager
def _path_lock(self, path):
    lockfile = self._join_path(path, 'lock')
    with fasteners.InterProcessLock(lockfile) as lock:
        with _storagefailure_wrapper():
            yield lock