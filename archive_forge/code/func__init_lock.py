from contextlib import contextmanager
import dbm
import os
import threading
from ..api import BytesBackend
from ..api import NO_VALUE
from ... import util
def _init_lock(self, argument, suffix, basedir, basefile, wrapper=None):
    if argument is None:
        lock = self.lock_factory(os.path.join(basedir, basefile + suffix))
    elif argument is not False:
        lock = self.lock_factory(os.path.abspath(os.path.normpath(argument)))
    else:
        return None
    if wrapper:
        lock = wrapper(lock)
    return lock