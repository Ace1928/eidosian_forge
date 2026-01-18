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
def _transaction(self):
    """This just wraps a global write-lock."""
    lock = self.backend.lock.write_lock
    with lock():
        yield